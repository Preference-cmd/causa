//! Independent external integration tests for the Slice 8 TurnContext edit API.
//!

use causa_kernel::{
    BlockContent, BlockId, BlockMeta, ContentPart, ContextBlock, EditError, Replacement,
    TextPayload, TurnContext, TurnId,
};
use std::ops::Range;

fn id(n: u128) -> BlockId {
    BlockId::new(uuid::Uuid::from_u128(n))
}

fn text_block(n: u128, text: &str) -> ContextBlock {
    ContextBlock::new(
        id(n),
        BlockContent::Parts(vec![ContentPart::Text(TextPayload::new(text))]),
        BlockMeta::default(),
    )
}

fn empty_parts_block(n: u128) -> ContextBlock {
    ContextBlock::new(id(n), BlockContent::Parts(Vec::new()), BlockMeta::default())
}

fn context(blocks: Vec<ContextBlock>) -> TurnContext {
    TurnContext::from_validated_blocks(TurnId::new("edit-test"), blocks).unwrap()
}

fn labels(ctx: &TurnContext) -> Vec<String> {
    labels_slice(ctx.blocks())
}

fn labels_slice(blocks: &[ContextBlock]) -> Vec<String> {
    blocks
        .iter()
        .map(|block| match block.content() {
            BlockContent::Parts(parts) => match parts.as_slice() {
                [ContentPart::Text(text)] => text.0.clone(),
                [] => "<empty-parts>".to_owned(),
                other => panic!("unexpected parts: {other:?}"),
            },
            other => panic!("unexpected content: {other:?}"),
        })
        .collect()
}

fn replacement(range: Range<usize>, with: Vec<ContextBlock>) -> Replacement {
    Replacement { range, with }
}

#[test]
fn replacements_use_the_original_sequence_and_append_to_the_final_tail() {
    let mut ctx = context(vec![
        text_block(1, "I"),
        text_block(2, "A"),
        text_block(3, "B"),
        text_block(4, "U"),
    ]);

    ctx.apply(
        vec![replacement(1..3, vec![text_block(10, "S")])],
        vec![text_block(11, "N1"), text_block(12, "N2")],
    )
    .unwrap();

    assert_eq!(labels(&ctx), ["I", "S", "U", "N1", "N2"]);

    // A preceding length-changing edit cannot shift a later original range.
    let mut ctx = context(vec![
        text_block(31, "I"),
        text_block(32, "A"),
        text_block(33, "B"),
        text_block(34, "C"),
        text_block(35, "U"),
    ]);
    ctx.apply(
        vec![
            replacement(1..4, vec![text_block(36, "S")]),
            replacement(2..3, vec![text_block(37, "T")]),
        ],
        Vec::new(),
    )
    .unwrap();
    assert_eq!(labels(&ctx), ["I", "A", "T", "C", "U"]);

    // A disjoint later range still selects original D after the first edit
    // changes the sequence length.
    let mut ctx = context(vec![
        text_block(41, "I"),
        text_block(42, "A"),
        text_block(43, "B"),
        text_block(44, "C"),
        text_block(45, "D"),
        text_block(46, "U"),
    ]);
    ctx.apply(
        vec![
            replacement(1..3, vec![text_block(47, "S")]),
            replacement(4..5, vec![text_block(48, "T")]),
        ],
        Vec::new(),
    )
    .unwrap();
    assert_eq!(labels(&ctx), ["I", "S", "C", "T", "U"]);

    // An explicit insertion at the original tail coexists with append, and
    // appears first because append is independent final-tail accumulation.
    let mut ctx = context(vec![
        text_block(21, "I"),
        text_block(22, "A"),
        text_block(23, "U"),
    ]);
    ctx.apply(
        vec![replacement(3..3, vec![text_block(24, "X")])],
        vec![text_block(25, "N")],
    )
    .unwrap();
    assert_eq!(labels(&ctx), ["I", "A", "U", "X", "N"]);
}

#[test]
fn later_overlaps_eliminate_whole_earlier_operations_without_revival() {
    let mut ctx = context(vec![
        text_block(1, "I"),
        text_block(2, "A"),
        text_block(3, "B"),
        text_block(4, "C"),
        text_block(5, "D"),
        text_block(6, "U"),
    ]);

    // The second operation eliminates the first; the third eliminates the
    // second. The first must not revive after its eliminator is eliminated.
    ctx.apply(
        vec![
            replacement(1..3, vec![text_block(10, "S")]),
            replacement(2..4, vec![text_block(11, "T")]),
            replacement(3..5, vec![text_block(12, "V")]),
        ],
        Vec::new(),
    )
    .unwrap();
    assert_eq!(labels(&ctx), ["I", "A", "B", "V", "U"]);
}

#[test]
fn insertion_points_overlap_on_left_and_inside_but_not_at_right_endpoint() {
    let base = || {
        vec![
            text_block(1, "I"),
            text_block(2, "A"),
            text_block(3, "B"),
            text_block(4, "U"),
        ]
    };

    let mut at_left = context(base());
    at_left
        .apply(
            vec![
                replacement(1..3, vec![text_block(10, "S")]),
                replacement(1..1, vec![text_block(11, "L")]),
            ],
            Vec::new(),
        )
        .unwrap();
    assert_eq!(labels(&at_left), ["I", "L", "A", "B", "U"]);

    let mut inside = context(base());
    inside
        .apply(
            vec![
                replacement(1..3, vec![text_block(20, "S")]),
                replacement(2..2, vec![text_block(21, "M")]),
            ],
            Vec::new(),
        )
        .unwrap();
    assert_eq!(labels(&inside), ["I", "A", "M", "B", "U"]);

    let mut at_right = context(base());
    at_right
        .apply(
            vec![
                replacement(1..3, vec![text_block(30, "S")]),
                replacement(3..3, vec![text_block(31, "R")]),
            ],
            Vec::new(),
        )
        .unwrap();
    assert_eq!(labels(&at_right), ["I", "S", "R", "U"]);

    // Half-open adjacent nonempty ranges do not overlap.
    let mut adjacent = context(base());
    adjacent
        .apply(
            vec![
                replacement(1..2, vec![text_block(40, "X")]),
                replacement(2..3, vec![text_block(41, "Y")]),
            ],
            Vec::new(),
        )
        .unwrap();
    assert_eq!(labels(&adjacent), ["I", "X", "Y", "U"]);
}

#[test]
fn empty_replacement_can_eliminate_same_point_insertion() {
    let mut ctx = context(vec![text_block(1, "I"), text_block(2, "A")]);
    ctx.apply(
        vec![
            replacement(1..1, vec![text_block(10, "must disappear")]),
            replacement(1..1, Vec::new()),
        ],
        Vec::new(),
    )
    .unwrap();
    assert_eq!(labels(&ctx), ["I", "A"]);
}

#[test]
fn invalid_ranges_are_checked_even_when_a_later_operation_covers_them() {
    let mut ctx = context(vec![
        text_block(1, "I"),
        text_block(2, "A"),
        text_block(3, "U"),
    ]);
    let before = labels(&ctx);
    let hidden = text_block(10, "hidden");
    let covering = text_block(11, "covering");

    let failure = ctx
        .apply(
            vec![
                replacement(1..99, vec![hidden.clone()]),
                replacement(1..2, vec![covering.clone()]),
            ],
            vec![text_block(12, "tail")],
        )
        .unwrap_err();

    assert!(matches!(
        failure.reason,
        EditError::InvalidRange {
            replacement_index: 0,
            range,
            len: 3,
        } if range == (1..99)
    ));
    assert_eq!(failure.replacements[0].range, 1..99);
    assert_eq!(failure.replacements[0].with, vec![hidden]);
    assert_eq!(failure.replacements[1].with, vec![covering]);
    assert_eq!(failure.appended, vec![text_block(12, "tail")]);
    assert_eq!(labels(&ctx), before);

    // A reversed range at a later original index is still rejected, and the
    // returned failure preserves every replacement and appended block.
    let first = text_block(13, "first");
    let reversed = text_block(14, "reversed");
    let later = text_block(15, "later");
    let appended = text_block(16, "append");
    let failure = ctx
        .apply(
            vec![
                replacement(0..1, vec![first.clone()]),
                replacement(Range { start: 2, end: 1 }, vec![reversed.clone()]),
                replacement(1..3, vec![later.clone()]),
            ],
            vec![appended.clone()],
        )
        .unwrap_err();
    assert!(matches!(
        failure.reason,
        EditError::InvalidRange {
            replacement_index: 1,
            range,
            len: 3,
        } if range == (Range { start: 2, end: 1 })
    ));
    assert_eq!(failure.replacements[0].with, vec![first]);
    assert_eq!(failure.replacements[1].with, vec![reversed]);
    assert_eq!(failure.replacements[2].with, vec![later]);
    assert_eq!(failure.appended, vec![appended]);
    assert_eq!(labels(&ctx), before);
}

#[test]
fn reusing_an_id_after_removal_requires_the_original_content_and_metadata() {
    let original = ContextBlock::new(
        id(2),
        BlockContent::Parts(vec![ContentPart::Text(TextPayload::new("A"))]),
        BlockMeta {
            source: Some("kept-source".into()),
            provider_call_id: None,
        },
    );
    let mut ctx = context(vec![
        text_block(1, "I"),
        original.clone(),
        text_block(3, "U"),
    ]);

    // Moving the same fact is allowed and preserves its identity.
    ctx.apply(
        vec![
            replacement(1..2, Vec::new()),
            replacement(3..3, vec![original.clone()]),
        ],
        Vec::new(),
    )
    .unwrap();
    assert_eq!(
        ctx.blocks()
            .iter()
            .map(ContextBlock::id)
            .collect::<Vec<_>>(),
        [id(1), id(3), id(2)]
    );

    // Removing the old occurrence does not permit rewriting its visible identity.
    let mut ctx = context(vec![
        text_block(1, "I"),
        original.clone(),
        text_block(3, "U"),
    ]);
    let changed_same_id = ContextBlock::new(
        id(2),
        BlockContent::Parts(vec![ContentPart::Text(TextPayload::new("changed"))]),
        original.meta().clone(),
    );
    let failure = ctx
        .apply(
            vec![replacement(1..2, vec![changed_same_id.clone()])],
            Vec::new(),
        )
        .unwrap_err();
    assert!(matches!(failure.reason, EditError::BlockIdentityMismatch(found) if found == id(2)));
    assert_eq!(failure.replacements[0].with, vec![changed_same_id]);
    assert_eq!(labels(&ctx), ["I", "A", "U"]);

    let changed_meta_same_id = ContextBlock::new(
        id(2),
        original.content().clone(),
        BlockMeta {
            source: Some("rewritten-source".into()),
            provider_call_id: None,
        },
    );
    let failure = ctx
        .apply(
            vec![replacement(1..2, vec![changed_meta_same_id])],
            Vec::new(),
        )
        .unwrap_err();
    assert!(matches!(failure.reason, EditError::BlockIdentityMismatch(found) if found == id(2)));
}

#[test]
fn ineffective_material_is_ignored_but_duplicate_final_ids_fail_atomically() {
    let base = || vec![text_block(1, "I"), text_block(2, "A"), text_block(3, "U")];

    let mut ctx = context(base());
    // The first operation carries a colliding ID, but is wholly eliminated by
    // the later operation; only effective replacement materials are validated.
    ctx.apply(
        vec![
            replacement(1..2, vec![text_block(2, "wrong identity payload")]),
            replacement(1..2, vec![text_block(10, "S")]),
        ],
        Vec::new(),
    )
    .unwrap();
    assert_eq!(labels(&ctx), ["I", "S", "U"]);

    let mut ctx = context(base());
    let duplicate_a = text_block(20, "X");
    let duplicate_b = text_block(20, "Y");
    let failure = ctx
        .apply(
            vec![replacement(
                1..2,
                vec![duplicate_a.clone(), duplicate_b.clone()],
            )],
            Vec::new(),
        )
        .unwrap_err();
    assert!(matches!(failure.reason, EditError::DuplicateBlockId(found) if found == id(20)));
    assert_eq!(failure.replacements[0].with, vec![duplicate_a, duplicate_b]);
    assert_eq!(labels(&ctx), ["I", "A", "U"]);

    // Explicitly fresh identities are accepted; an empty Parts block remains a
    // real member and is distinct from deleting the range.
    let mut ctx = context(base());
    ctx.apply(
        vec![replacement(1..2, vec![empty_parts_block(21)])],
        Vec::new(),
    )
    .unwrap();
    assert_eq!(labels(&ctx), ["I", "<empty-parts>", "U"]);
}

#[test]
fn builder_reads_original_blocks_and_drop_does_not_commit() {
    let mut ctx = context(vec![
        text_block(1, "I"),
        text_block(2, "A"),
        text_block(3, "U"),
    ]);
    let held = ctx.blocks()[1..2].to_vec();
    {
        let _dropped = ctx.edit().replace(1..2, vec![text_block(10, "S")]);
    }
    assert_eq!(labels(&ctx), ["I", "A", "U"]);
    assert_eq!(held, vec![text_block(2, "A")]);

    let edit = ctx
        .edit()
        .replace(1..2, vec![text_block(11, "S")])
        .append(vec![text_block(12, "tail")]);
    assert_eq!(labels_slice(edit.blocks()), ["I", "A", "U"]);
    edit.commit().unwrap();
    assert_eq!(labels(&ctx), ["I", "S", "U", "tail"]);
}

#[test]
fn sealed_check_has_priority_over_range_errors_and_rejects_empty_commit() {
    let mut ctx = context(vec![text_block(1, "I")]);
    ctx.seal();

    let failure = ctx
        .apply(
            vec![replacement(0..9, vec![text_block(2, "bad range")])],
            Vec::new(),
        )
        .unwrap_err();
    assert!(matches!(failure.reason, EditError::SealedTurn));
    assert_eq!(failure.replacements[0].range, 0..9);
    assert_eq!(
        failure.replacements[0].with,
        vec![text_block(2, "bad range")]
    );
    assert_eq!(labels(&ctx), ["I"]);

    let empty_failure = ctx.apply(Vec::new(), Vec::new()).unwrap_err();
    assert!(matches!(empty_failure.reason, EditError::SealedTurn));
    assert!(empty_failure.replacements.is_empty());
    assert!(empty_failure.appended.is_empty());
}

#[test]
fn moved_payloads_keep_their_string_allocations_during_successful_apply() {
    let retained_text = "retained original payload with enough bytes to heap allocate".to_owned();
    let replacement_text =
        "owned replacement payload with enough bytes to heap allocate".to_owned();
    let retained_ptr = retained_text.as_ptr();
    let replacement_ptr = replacement_text.as_ptr();

    let retained = ContextBlock::new(
        id(1),
        BlockContent::Parts(vec![ContentPart::Text(TextPayload(retained_text))]),
        BlockMeta::default(),
    );
    let replaced = text_block(2, "remove me");
    let mut ctx = context(vec![retained, replaced]);
    let incoming = ContextBlock::new(
        id(3),
        BlockContent::Parts(vec![ContentPart::Text(TextPayload(replacement_text))]),
        BlockMeta::default(),
    );

    ctx.apply(vec![replacement(1..2, vec![incoming])], Vec::new())
        .unwrap();

    let get_ptr = |block: &ContextBlock| match block.content() {
        BlockContent::Parts(parts) => match &parts[0] {
            ContentPart::Text(text) => text.0.as_ptr(),
            other => panic!("unexpected part: {other:?}"),
        },
        other => panic!("unexpected content: {other:?}"),
    };
    assert_eq!(get_ptr(&ctx.blocks()[0]), retained_ptr);
    assert_eq!(get_ptr(&ctx.blocks()[1]), replacement_ptr);
}

#[derive(Clone, Copy, Debug)]
struct OracleOp {
    start: usize,
    end: usize,
    has_material: bool,
    material_id: u128,
}

fn footprint(op: OracleOp) -> u128 {
    if op.start == op.end {
        1u128 << (2 * op.start)
    } else {
        (2 * op.start..2 * op.end).fold(0, |mask, bit| mask | (1u128 << bit))
    }
}

fn oracle_result(base: &[String], ops: &[OracleOp]) -> Vec<String> {
    // Each original gap and slot has its own bit: gap i -> bit 2*i, slot i ->
    // bit 2*i+1. A zero-width edit touches just its gap; [s,e) touches the
    // alternating gaps and original slots from gap s up to, but not including,
    // gap e. This geometry determines conflict without interval comparisons.
    assert!(base.len() < 64, "the small oracle uses a u128 footprint");
    let masks: Vec<u128> = ops.iter().copied().map(footprint).collect();
    let mut alive = Vec::<usize>::new();
    for index in 0..ops.len() {
        alive.retain(|prior| masks[*prior] & masks[index] == 0);
        alive.push(index);
    }

    let mut starts = vec![None; base.len() + 1];
    let mut deleted_slots = vec![false; base.len()];
    for index in alive {
        let op = ops[index];
        assert!(starts[op.start].replace(index).is_none());
        for deleted in &mut deleted_slots[op.start..op.end] {
            *deleted = true;
        }
    }

    let mut result = Vec::new();
    for gap in 0..=base.len() {
        if let Some(index) = starts[gap] {
            let op = ops[index];
            if op.has_material {
                result.push(format!("M{}", op.material_id));
            }
        }
        if gap < base.len() && !deleted_slots[gap] {
            result.push(base[gap].clone());
        }
    }
    result
}

#[test]
fn slot_gap_oracle_preserves_original_slots_around_zero_width_edits() {
    let base = vec!["S0".to_owned(), "S1".to_owned(), "S2".to_owned()];
    let insertion_at_start = OracleOp {
        start: 0,
        end: 0,
        has_material: true,
        material_id: 201,
    };
    let insertion_inside = OracleOp {
        start: 1,
        end: 1,
        has_material: true,
        material_id: 202,
    };
    let insertion_at_tail = OracleOp {
        start: 3,
        end: 3,
        has_material: true,
        material_id: 203,
    };
    let empty_insertion = OracleOp {
        start: 2,
        end: 2,
        has_material: false,
        material_id: 204,
    };

    assert_eq!(
        oracle_result(&base, &[insertion_at_start]),
        ["M201", "S0", "S1", "S2"]
    );
    assert_eq!(
        oracle_result(&base, &[insertion_inside]),
        ["S0", "M202", "S1", "S2"]
    );
    assert_eq!(
        oracle_result(&base, &[insertion_at_tail]),
        ["S0", "S1", "S2", "M203"]
    );
    assert_eq!(oracle_result(&base, &[empty_insertion]), base);
}

#[test]
fn exhaustive_three_operation_batches_match_an_independent_slot_gap_oracle() {
    let base_labels = vec!["S0".to_owned(), "S1".to_owned(), "S2".to_owned()];
    let ranges = [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 2),
        (1, 3),
        (0, 3),
    ];
    let choices: Vec<OracleOp> = ranges
        .into_iter()
        .flat_map(|(start, end)| [false, true].map(move |has_material| (start, end, has_material)))
        .enumerate()
        .map(|(ordinal, (start, end, has_material))| OracleOp {
            start,
            end,
            has_material,
            material_id: 100 + ordinal as u128,
        })
        .collect();

    for first in choices.iter().copied() {
        for second in choices.iter().copied() {
            for third in choices.iter().copied() {
                let ops = [first, second, third];
                let expected = oracle_result(&base_labels, &ops);
                let mut ctx = context(
                    base_labels
                        .iter()
                        .enumerate()
                        .map(|(i, text)| text_block((i + 1) as u128, text))
                        .collect(),
                );
                let replacements = ops
                    .iter()
                    .map(|op| {
                        replacement(
                            op.start..op.end,
                            if op.has_material {
                                vec![text_block(op.material_id, &format!("M{}", op.material_id))]
                            } else {
                                Vec::new()
                            },
                        )
                    })
                    .collect();
                ctx.apply(replacements, Vec::new()).unwrap();
                assert_eq!(labels(&ctx), expected, "operations: {ops:?}");
            }
        }
    }
}

fn tagged_block(n: u128, text: &str, source: &str) -> ContextBlock {
    ContextBlock::new(
        id(n),
        BlockContent::Parts(vec![ContentPart::Text(TextPayload::new(text))]),
        BlockMeta {
            source: Some(source.to_owned()),
            provider_call_id: Some(format!("provider-{n}")),
        },
    )
}

fn assert_failure_payload(
    failure: causa_kernel::EditFailure,
    reason: EditError,
    replacements: &[Replacement],
    appended: &[ContextBlock],
) {
    assert_eq!(failure.reason, reason);
    assert_eq!(failure.replacements, replacements);
    assert_eq!(failure.appended, appended);
}

fn assert_wire_unchanged(ctx: &TurnContext, before: &serde_json::Value) {
    assert_eq!(&serde_json::to_value(ctx).unwrap(), before);
}

#[test]
fn all_edit_errors_return_every_input_and_leave_context_wire_unchanged() {
    // Sealed is first and also applies through the builder commit path.
    let mut sealed = context(vec![tagged_block(1, "original", "source-original")]);
    sealed.seal();
    let before = serde_json::to_value(&sealed).unwrap();
    let sealed_replacements = vec![replacement(
        Range { start: 0, end: 9 },
        vec![tagged_block(10, "sealed replacement", "sealed-source")],
    )];
    let sealed_appended = vec![tagged_block(11, "sealed append", "append-source")];
    let failure = sealed
        .edit()
        .replace(
            sealed_replacements[0].range.clone(),
            sealed_replacements[0].with.clone(),
        )
        .append(sealed_appended.clone())
        .commit()
        .unwrap_err();
    assert_failure_payload(
        failure,
        EditError::SealedTurn,
        &sealed_replacements,
        &sealed_appended,
    );
    assert_wire_unchanged(&sealed, &before);

    // Every original range is checked, including one hidden by the later edit.
    let mut invalid = context(vec![
        tagged_block(20, "A", "source-A"),
        tagged_block(21, "B", "source-B"),
    ]);
    let before = serde_json::to_value(&invalid).unwrap();
    let invalid_replacements = vec![
        replacement(
            0..1,
            vec![tagged_block(22, "will be covered", "covered-source")],
        ),
        replacement(
            Range { start: 2, end: 1 },
            vec![tagged_block(23, "reversed", "reversed-source")],
        ),
        replacement(
            0..2,
            vec![tagged_block(24, "later covering edit", "cover-source")],
        ),
    ];
    let invalid_appended = vec![tagged_block(25, "tail", "tail-source")];
    let failure = invalid
        .apply(invalid_replacements.clone(), invalid_appended.clone())
        .unwrap_err();
    assert_failure_payload(
        failure,
        EditError::InvalidRange {
            replacement_index: 1,
            range: Range { start: 2, end: 1 },
            len: 2,
        },
        &invalid_replacements,
        &invalid_appended,
    );
    assert_wire_unchanged(&invalid, &before);

    // Reusing a removed original ID with changed metadata still fails, and the
    // complete original replacement and append materials are returned.
    let mut mismatch = context(vec![tagged_block(30, "original", "source-original")]);
    let before = serde_json::to_value(&mismatch).unwrap();
    let mismatch_replacements = vec![replacement(
        0..1,
        vec![tagged_block(30, "original", "changed-source")],
    )];
    let mismatch_appended = vec![tagged_block(31, "append", "append-source")];
    let failure = mismatch
        .apply(mismatch_replacements.clone(), mismatch_appended.clone())
        .unwrap_err();
    assert_failure_payload(
        failure,
        EditError::BlockIdentityMismatch(id(30)),
        &mismatch_replacements,
        &mismatch_appended,
    );
    assert_wire_unchanged(&mismatch, &before);

    // Duplicate detection uses the final members but returns all material.
    let mut duplicate = context(vec![tagged_block(40, "original", "source-original")]);
    let before = serde_json::to_value(&duplicate).unwrap();
    let duplicate_replacements = vec![replacement(
        1..1,
        vec![
            tagged_block(41, "duplicate one", "first-source"),
            tagged_block(41, "duplicate two", "second-source"),
        ],
    )];
    let duplicate_appended = vec![tagged_block(42, "append", "append-source")];
    let failure = duplicate
        .apply(duplicate_replacements.clone(), duplicate_appended.clone())
        .unwrap_err();
    assert_failure_payload(
        failure,
        EditError::DuplicateBlockId(id(41)),
        &duplicate_replacements,
        &duplicate_appended,
    );
    assert_wire_unchanged(&duplicate, &before);
}

#[test]
fn open_context_accepts_empty_apply_and_builder_commit_without_wire_changes() {
    let mut ctx = context(vec![tagged_block(50, "kept", "kept-source")]);
    let before = serde_json::to_value(&ctx).unwrap();

    ctx.apply(Vec::new(), Vec::new()).unwrap();
    assert_wire_unchanged(&ctx, &before);

    ctx.edit().commit().unwrap();
    assert_wire_unchanged(&ctx, &before);
}

#[test]
fn identity_mismatch_precedes_duplicate_final_id_validation() {
    let mut ctx = context(vec![
        tagged_block(1, "prefix", "prefix-source"),
        tagged_block(2, "original", "original-source"),
    ]);
    let before = serde_json::to_value(&ctx).unwrap();
    let replacements = vec![replacement(
        1..2,
        vec![
            tagged_block(10, "duplicate first", "first-source"),
            tagged_block(10, "duplicate second", "second-source"),
        ],
    )];
    let appended = vec![tagged_block(2, "rewritten original", "changed-source")];

    let failure = ctx
        .apply(replacements.clone(), appended.clone())
        .unwrap_err();
    assert_failure_payload(
        failure,
        EditError::BlockIdentityMismatch(id(2)),
        &replacements,
        &appended,
    );
    assert_wire_unchanged(&ctx, &before);
}
