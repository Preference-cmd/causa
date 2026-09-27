mod common;

use causa_kernel::{
    BatchError, BlockId, TextPayload, ToolBatch, ToolCallContext, ToolCallId, ToolCallPayload,
    ToolOutput, ToolResultPayload, ToolResultStatus,
};
use serde_json::{Value, json};
use std::collections::HashSet;

fn call(id: BlockId, name: &str, arguments: Value) -> ToolCallContext {
    ToolCallContext::from_declaration(
        id,
        &ToolCallPayload {
            tool_name: name.into(),
            arguments,
        },
    )
}

fn result(call_block_id: BlockId, note: &str) -> ToolResultPayload {
    ToolResultPayload {
        call_block_id,
        status: ToolResultStatus::Succeeded,
        output: ToolOutput::new(json!({"ok": true})),
        media: Vec::new(),
        notes: vec![TextPayload::new(note)],
    }
}

#[test]
fn borrowed_call_key_uses_full_value_equality_and_hashing() {
    let args_a = json!({"nested": {"x": 1, "y": [true, null]}, "n": 4});
    let args_b = json!({"n": 4, "nested": {"y": [true, null], "x": 1}});
    let args_c = json!({"nested": {"x": 2, "y": [true, null]}, "n": 4});
    let first = ToolCallId::new("search", &args_a);
    let reordered = ToolCallId::new("search", &args_b);
    let changed = ToolCallId::new("search", &args_c);
    assert_eq!(first, reordered);
    assert_ne!(first, changed);
    let mut keys = HashSet::new();
    keys.insert(first);
    keys.insert(reordered);
    keys.insert(changed);
    assert_eq!(keys.len(), 2);

    let integer = json!(1);
    let float = json!(1.0);
    let integer_key = ToolCallId::new("number", &integer);
    let float_key = ToolCallId::new("number", &float);
    let mut numeric_keys = HashSet::new();
    numeric_keys.insert(integer_key);
    numeric_keys.insert(float_key);
    assert_eq!(
        numeric_keys.len(),
        usize::from(integer_key != float_key) + 1
    );
}

#[test]
fn batch_resolves_by_declaration_and_merges_notes_once() {
    let declaration_a = common::block_id();
    let declaration_b = common::block_id();
    let mut batch = ToolBatch::new(vec![
        call(declaration_a, "same", json!({"q": 1})),
        call(declaration_b, "same", json!({"q": 1})),
    ])
    .unwrap();
    assert_eq!(batch.calls().len(), 2);
    batch.calls_mut()[0].push_note(TextPayload::new("preflight"));
    let result_b_id = common::block_id();
    batch
        .resolve_at(1, result_b_id, result(declaration_b, "tool note"))
        .unwrap();
    assert_eq!(batch.completed_len(), 1);
    assert_eq!(batch.results()[0].call().call_block_id, declaration_b);
    assert_eq!(batch.calls()[0].call().call_block_id, declaration_a);
    assert_eq!(batch.results()[0].result().unwrap().0, &result_b_id);
    assert_eq!(
        batch.results()[0]
            .result()
            .unwrap()
            .1
            .notes
            .iter()
            .map(|note| note.0.as_str())
            .collect::<Vec<_>>(),
        ["tool note"]
    );

    // The pending call keeps its own declaration identity even though its
    // content key matches the completed sibling.
    let result_a_id = common::block_id();
    batch
        .resolve_at(1, result_a_id, result(declaration_a, "later"))
        .unwrap();
    assert!(batch.calls().is_empty());
    assert_eq!(batch.completed_len(), 2);
    assert_eq!(batch.results()[0].call().call_block_id, declaration_b);
    assert_eq!(batch.results()[1].call().call_block_id, declaration_a);
    assert_eq!(
        batch.results()[1].result().unwrap().1.notes[0].0,
        "preflight"
    );
    assert_eq!(batch.results()[1].result().unwrap().1.notes[1].0, "later");
    batch.validate().unwrap();
}

#[test]
fn resolve_validation_is_atomic_and_results_keep_supplied_commit_order() {
    let declaration_a = common::block_id();
    let declaration_b = common::block_id();
    let wrong_call = common::block_id();
    let mut batch = ToolBatch::new(vec![
        call(declaration_a, "a", json!({})),
        call(declaration_b, "b", json!({})),
    ])
    .unwrap();
    let before = batch.completed_len();
    let bad_id = common::block_id();
    assert!(matches!(
        batch.resolve_at(0, bad_id, result(wrong_call, "bad")),
        Err(BatchError::MismatchedCall { .. })
    ));
    assert_eq!(batch.completed_len(), before);
    assert!(batch.calls()[0].result().is_none());

    let result_b_id = common::block_id();
    let result_a_id = common::block_id();
    batch
        .resolve_at(1, result_b_id, result(declaration_b, "b"))
        .unwrap();
    batch
        .resolve_at(1, result_a_id, result(declaration_a, "a"))
        .unwrap();
    let committed = batch.into_results().unwrap();
    assert_eq!(committed[0].0, result_b_id);
    assert_eq!(committed[0].1.call_block_id, declaration_b);
    assert_eq!(committed[1].0, result_a_id);
    assert_eq!(committed[1].1.call_block_id, declaration_a);
}

#[test]
fn validate_detects_result_id_collision_introduced_by_cross_batch_swap() {
    let declaration_a = common::block_id();
    let declaration_x = common::block_id();
    let declaration_y = common::block_id();
    let declaration_z = common::block_id();

    let mut batch_a = ToolBatch::new(vec![call(declaration_a, "a", json!({}))]).unwrap();
    batch_a
        .resolve_at(0, declaration_x, result(declaration_a, "a"))
        .unwrap();

    let mut batch_b = ToolBatch::new(vec![
        call(declaration_x, "x", json!({})),
        call(declaration_y, "y", json!({})),
        call(declaration_z, "z", json!({})),
    ])
    .unwrap();
    batch_b
        .resolve_at(0, common::block_id(), result(declaration_x, "x"))
        .unwrap();
    batch_b
        .resolve_at(1, common::block_id(), result(declaration_y, "y"))
        .unwrap();
    batch_b
        .resolve_at(2, common::block_id(), result(declaration_z, "z"))
        .unwrap();

    std::mem::swap(&mut batch_a.results_mut()[0], &mut batch_b.results_mut()[1]);
    assert!(matches!(
        batch_b.validate(),
        Err(BatchError::DuplicateResultBlockId(id)) if id == declaration_x
    ));
    assert!(matches!(
        batch_b.resolve_at(0, common::block_id(), result(declaration_x, "retry")),
        Err(BatchError::DuplicateResultBlockId(id)) if id == declaration_x
    ));
    assert_eq!(batch_b.completed_len(), 3);
}

#[test]
fn empty_batch_and_middle_completion_use_native_slice_order() {
    let empty = ToolBatch::new(Vec::new()).unwrap();
    assert_eq!(empty.completed_len(), 0);
    assert!(empty.into_results().unwrap().is_empty());
    let ids = [
        common::block_id(),
        common::block_id(),
        common::block_id(),
        common::block_id(),
    ];
    let mut batch =
        ToolBatch::new(ids.iter().map(|id| call(*id, "same", json!({}))).collect()).unwrap();
    batch
        .resolve_at(2, common::block_id(), result(ids[2], "c"))
        .unwrap();
    assert_eq!(
        batch.declaration_ids(),
        vec![ids[2], ids[1], ids[0], ids[3]]
    );
    batch.calls_mut().reverse();
    assert_eq!(
        batch.declaration_ids(),
        vec![ids[2], ids[3], ids[0], ids[1]]
    );
    let before = format!("{batch:?}");
    for index in [0, 4, usize::MAX] {
        assert!(matches!(
            batch.resolve_at(index, common::block_id(), result(ids[0], "invalid")),
            Err(BatchError::InvalidIndex { .. })
        ));
        assert_eq!(format!("{batch:?}"), before);
    }
    assert!(matches!(
        batch.resolve_at(1, ids[0], result(ids[3], "collision")),
        Err(BatchError::DuplicateResultBlockId(_))
    ));
    assert_eq!(format!("{batch:?}"), before);
    let (error, returned) = batch.into_results().unwrap_err();
    assert!(matches!(
        error,
        BatchError::Incomplete {
            completed: 1,
            total: 4
        }
    ));
    assert_eq!(format!("{returned:?}"), before);
}

#[test]
fn malformed_partition_does_not_panic_or_consume_notes() {
    let a = common::block_id();
    let b = common::block_id();
    let mut pending =
        ToolBatch::new(vec![call(a, "a", json!({})), call(b, "b", json!({}))]).unwrap();
    pending.calls_mut()[1].push_note(TextPayload::new("keep me"));
    let c = common::block_id();
    let mut completed = ToolBatch::new(vec![call(c, "c", json!({}))]).unwrap();
    completed
        .resolve_at(0, common::block_id(), result(c, "done"))
        .unwrap();
    std::mem::swap(&mut pending.calls_mut()[0], &mut completed.results_mut()[0]);
    let before = format!("{pending:?}");
    assert!(matches!(
        pending.resolve_at(1, common::block_id(), result(b, "new")),
        Err(BatchError::InvalidPartition(_))
    ));
    assert_eq!(format!("{pending:?}"), before);
}

#[test]
fn numeric_content_keys_preserve_value_semantics_at_precision_boundaries() {
    use std::hash::{DefaultHasher, Hash, Hasher};
    fn hash(key: ToolCallId<'_>) -> u64 {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }
    let values = [
        json!(0),
        json!(0.0),
        json!(-0.0),
        json!(1),
        json!(1.0),
        json!(9_007_199_254_740_992_u64),
        json!(9_007_199_254_740_993_u64),
        json!(u64::MAX),
        json!(i64::MIN),
    ];
    for a in &values {
        for b in &values {
            let ka = ToolCallId::new("number", a);
            let kb = ToolCallId::new("number", b);
            assert_eq!(ka == kb, a == b);
            if ka == kb {
                assert_eq!(hash(ka), hash(kb));
            }
        }
    }
    assert_ne!(
        ToolCallId::new("number", &values[5]),
        ToolCallId::new("number", &values[6])
    );
    assert_ne!(
        ToolCallId::new("number", &values[0]),
        ToolCallId::new("other", &values[0])
    );
}

#[test]
fn completion_merges_pending_notes_once_for_every_terminal_status() {
    for status in [
        ToolResultStatus::Succeeded,
        ToolResultStatus::Failed,
        ToolResultStatus::Rejected,
        ToolResultStatus::Cancelled,
        ToolResultStatus::TimedOut,
        ToolResultStatus::UnknownOutcome,
    ] {
        let id = common::block_id();
        let mut batch = ToolBatch::new(vec![call(id, "work", json!({"limit":20}))]).unwrap();
        batch.calls_mut()[0].input_mut().unwrap().arguments["limit"] = json!(5);
        batch.calls_mut()[0].push_note(TextPayload::new("limit is five"));
        let mut output = result(id, "tool detail");
        output.status = status.clone();
        batch
            .resolve_at(0, common::block_id(), output.clone())
            .unwrap();
        assert!(batch.resolve_at(0, common::block_id(), output).is_err());
        batch.results_mut()[0].push_note(TextPayload::new("post detail"));
        let entry = &batch.results()[0];
        assert_eq!(entry.call().input.arguments["limit"], 5);
        assert!(entry.call().result_notes.is_empty());
        let (_, result) = entry.result().unwrap();
        assert_eq!(result.status, status);
        assert_eq!(
            result.notes,
            [
                TextPayload::new("limit is five"),
                TextPayload::new("tool detail"),
                TextPayload::new("post detail")
            ]
        );
    }
}
