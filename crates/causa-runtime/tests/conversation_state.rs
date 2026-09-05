//! Session-aggregate acceptance — facts and order. Graduated from the
//! kernel's conversation suite in Slice 6.5 together with the aggregate
//! itself (`ConversationState` is runtime vocabulary now): commit/seal/
//! abort discipline, merged views, validated replay, and the paused-state
//! round-trip. The runner-entry coverage lives in
//! `conversation_entries.rs`; these tests play the host's stamping role
//! via `seal_turn` directly.

mod common;

use common::commit_sealed;

use causa_kernel::{
    ContextVersion, ConversationId, FrameScope, OrderedBlocks, RoundId, TextPayload, TurnId,
    TurnSnapshot,
};
use causa_runtime::{
    ConversationError, ConversationState, HistoryEntry, SealedResult, TurnSequence,
};

fn conv() -> ConversationState {
    ConversationState::new(ConversationId("conv-1".into()))
}

/// Backward-compat shim: keep the historical test call sites readable.
fn commit_completed(state: &mut ConversationState, turn_id: &str) {
    commit_sealed(state, turn_id, SealedResult::Completed);
}

#[test]
fn commit_is_exactly_once_and_assigns_sequence() {
    let mut c = conv();
    c.begin_turn(TurnId::new("t1")).unwrap();
    c.active_turn_mut()
        .unwrap()
        .append_input(TextPayload::new("hi"), "user")
        .unwrap();
    c.seal_turn(TurnId::new("t1"), SealedResult::Completed)
        .unwrap();
    let entry = c.commit(TurnId::new("t1")).unwrap();
    assert_eq!(entry.sequence.0, 0);
    assert_eq!(entry.snapshot.turn_id.0, "t1");
    // The snapshot itself carries no session order (Slice 6.5): the
    // entry is the only place a sequence exists.
    let wire = serde_json::to_string(&entry.snapshot).unwrap();
    assert!(!wire.contains("turn_sequence"), "{wire}");
    // Repeated commit: the active slot is empty, so rejection lands on
    // UnknownTurn — rejection, not idempotence.
    assert!(matches!(
        c.commit(TurnId::new("t1")),
        Err(ConversationError::UnknownTurn(_))
    ));
    assert_eq!(c.history_len(), 1);
    assert!(c.active_turn().is_none());
    assert_eq!(c.version().0, 2); // begin + commit
}

#[test]
fn interrupted_turn_never_enters_history_and_id_is_reusable() {
    let mut c = conv();
    commit_completed(&mut c, "t1");
    c.begin_turn(TurnId::new("t2")).unwrap();
    c.seal_turn(TurnId::new("t2"), SealedResult::Interrupted)
        .unwrap();
    // Interrupted stamp: commit rejects.
    assert!(matches!(
        c.commit(TurnId::new("t2")),
        Err(ConversationError::TurnNotCompleted(_))
    ));
    // Abort discards in any state; history untouched; id reusable.
    let aborted = c.abort_turn(TurnId::new("t2")).unwrap();
    assert!(aborted.is_sealed());
    assert_eq!(c.history_len(), 1);
    c.begin_turn(TurnId::new("t2")).unwrap();
}

#[test]
fn begin_turn_rejects_active_and_duplicate_ids() {
    let mut c = conv();
    c.begin_turn(TurnId::new("t1")).unwrap();
    // Open active in the slot.
    assert!(matches!(
        c.begin_turn(TurnId::new("t2")),
        Err(ConversationError::TurnAlreadyActive)
    ));
    // Sealed active in the slot — different variant.
    c.seal_turn(TurnId::new("t1"), SealedResult::Completed)
        .unwrap();
    assert!(matches!(
        c.begin_turn(TurnId::new("t2")),
        Err(ConversationError::TurnAlreadySealed)
    ));
    // After commit, the id collides with committed history.
    c.commit(TurnId::new("t1")).unwrap();
    assert!(matches!(
        c.begin_turn(TurnId::new("t1")),
        Err(ConversationError::DuplicateTurnId(_))
    ));
}

#[test]
fn unknown_turn_rejected_across_operations() {
    let mut c = conv();
    c.begin_turn(TurnId::new("t1")).unwrap();
    assert!(matches!(
        c.seal_turn(TurnId::new("nope"), SealedResult::Completed),
        Err(ConversationError::UnknownTurn(_))
    ));
    assert!(matches!(
        c.commit(TurnId::new("nope")),
        Err(ConversationError::UnknownTurn(_))
    ));
    assert!(matches!(
        c.abort_turn(TurnId::new("nope")),
        Err(ConversationError::UnknownTurn(_))
    ));
}

#[test]
fn commit_rejects_open_turn() {
    let mut c = conv();
    c.begin_turn(TurnId::new("t1")).unwrap();
    // Not sealed, no stamp — TurnNotCompleted, not a panic.
    assert!(matches!(
        c.commit(TurnId::new("t1")),
        Err(ConversationError::TurnNotCompleted(_))
    ));
}

#[test]
fn version_ticks_on_begin_commit_abort_only() {
    let mut c = conv();
    assert_eq!(c.version().0, 0);
    c.begin_turn(TurnId::new("t1")).unwrap();
    assert_eq!(c.version().0, 1);
    // seal_turn records the outcome; it is not a version transition.
    c.seal_turn(TurnId::new("t1"), SealedResult::Completed)
        .unwrap();
    assert_eq!(c.version().0, 1);
    c.commit(TurnId::new("t1")).unwrap();
    assert_eq!(c.version().0, 2);
    c.begin_turn(TurnId::new("t2")).unwrap();
    assert_eq!(c.version().0, 3);
    c.seal_turn(TurnId::new("t2"), SealedResult::Interrupted)
        .unwrap();
    assert_eq!(c.version().0, 3);
    c.abort_turn(TurnId::new("t2")).unwrap();
    assert_eq!(c.version().0, 4);
}

#[test]
fn conversations_are_fully_isolated() {
    let mut a = ConversationState::new(ConversationId("A".into()));
    let mut b = ConversationState::new(ConversationId("B".into()));
    commit_completed(&mut a, "shared-id");
    assert_eq!(a.history_len(), 1);
    assert_eq!(b.history_len(), 0);
    assert_eq!(b.version().0, 0);
    // The same turn id lives independently in each conversation.
    commit_completed(&mut b, "shared-id");
    assert_eq!(b.history_len(), 1);
    assert_eq!(b.history()[0].sequence.0, 0);
}

#[test]
fn committed_snapshot_carries_turn_facts_and_abort_leaves_history() {
    let mut c = conv();
    c.begin_turn(TurnId::new("t1")).unwrap();
    c.active_turn_mut()
        .unwrap()
        .append_input(TextPayload::new("hello"), "user")
        .unwrap();
    c.seal_turn(TurnId::new("t1"), SealedResult::Completed)
        .unwrap();
    let entry = c.commit(TurnId::new("t1")).unwrap();
    assert_eq!(entry.snapshot.blocks.as_slice().len(), 1);
    // Aborting a later turn leaves committed history byte-identical.
    c.begin_turn(TurnId::new("t2")).unwrap();
    c.seal_turn(TurnId::new("t2"), SealedResult::Interrupted)
        .unwrap();
    c.abort_turn(TurnId::new("t2")).unwrap();
    assert_eq!(c.history_len(), 1);
    assert_eq!(c.history()[0].snapshot.turn_id.0, "t1");
}

// ---- lossless merged view ------------------------------------------------------

/// Two committed turns (one input block each) plus nothing active.
fn build_two_turn_history() -> ConversationState {
    let mut c = ConversationState::new(ConversationId("conv-1".into()));
    for tid in ["t1", "t2"] {
        c.begin_turn(TurnId::new(tid)).unwrap();
        c.active_turn_mut()
            .unwrap()
            .append_input(TextPayload::new(format!("in-{tid}")), "user")
            .unwrap();
        c.seal_turn(TurnId::new(tid), SealedResult::Completed)
            .unwrap();
        c.commit(TurnId::new(tid)).unwrap();
    }
    c
}

#[test]
fn merged_frame_orders_history_then_active_under_conversation_scope() {
    let mut c = build_two_turn_history();
    c.begin_turn(TurnId::new("t3")).unwrap();
    c.active_turn_mut()
        .unwrap()
        .append_input(TextPayload::new("in-t3"), "user")
        .unwrap();
    let f = c.frame(RoundId(0)).unwrap();
    // history blocks in sequence order, then active blocks.
    let blocks = &f.model_context.blocks;
    assert_eq!(blocks.len(), 3);
    assert_eq!(blocks[0].id.turn_id.0, "t1");
    assert_eq!(blocks[1].id.turn_id.0, "t2");
    assert_eq!(blocks[2].id.turn_id.0, "t3");
    // Conversation scope identity: active turn's version is the pin.
    match &f.scope {
        FrameScope::Conversation {
            conversation_id,
            active_turn_id,
            source_version,
        } => {
            assert_eq!(conversation_id.0, "conv-1");
            assert_eq!(active_turn_id.0, "t3");
            assert_eq!(source_version.0, 1);
        }
        _ => panic!("expected conversation scope"),
    }
    // Deterministic per (conversation, active turn, source version, round).
    assert_eq!(f.frame_id, c.frame(RoundId(0)).unwrap().frame_id);
    assert_ne!(f.frame_id, c.frame(RoundId(1)).unwrap().frame_id);
}

#[test]
fn empty_history_frame_matches_turn_projection_block_for_block() {
    let mut c = ConversationState::new(ConversationId("conv-1".into()));
    c.begin_turn(TurnId::new("t1")).unwrap();
    c.active_turn_mut()
        .unwrap()
        .append_input(TextPayload::new("hi"), "user")
        .unwrap();
    let merged = c.frame(RoundId(0)).unwrap();
    let single = c.active_turn().unwrap().frame(RoundId(0));
    // Byte-equal content, different identity: the scopes differ
    // (Conversation vs Turn), so frame ids must differ too.
    assert_eq!(
        serde_json::to_string(&merged.model_context.blocks).unwrap(),
        serde_json::to_string(&single.model_context.blocks).unwrap()
    );
    assert_ne!(merged.frame_id, single.frame_id);
}

#[test]
fn merged_frame_is_lossless_and_writes_nothing_back() {
    let mut c = build_two_turn_history();
    c.begin_turn(TurnId::new("t3")).unwrap();
    c.active_turn_mut()
        .unwrap()
        .append_input(TextPayload::new("in-t3"), "user")
        .unwrap();
    let before_history = serde_json::to_string(c.history()).unwrap();
    let before_active = serde_json::to_string(&c.active_turn().unwrap().snapshot_blocks()).unwrap();
    let _ = c.frame(RoundId(0)).unwrap();
    assert_eq!(serde_json::to_string(c.history()).unwrap(), before_history);
    assert_eq!(
        serde_json::to_string(&c.active_turn().unwrap().snapshot_blocks()).unwrap(),
        before_active
    );
    // frame() is a projection, not a transition: two commits + begin = 5.
    assert_eq!(c.version().0, 5);
}

#[test]
fn frame_without_active_turn_is_rejected() {
    let c = build_two_turn_history();
    // Pure-history inspection goes through history(); the merged frame
    // needs an active turn for its scope identity.
    assert!(matches!(
        c.frame(RoundId(0)),
        Err(ConversationError::NoActiveTurn)
    ));
}

// ---- validated replay ----------------------------------------------------------

/// Out-of-order and duplicate turn sequences are rejected.
#[test]
fn from_history_rejects_non_monotonic_sequences() {
    let live = build_two_turn_history();
    // Out of order: [seq 1, seq 0].
    let swapped: Vec<_> = live.history().iter().rev().cloned().collect();
    assert!(matches!(
        ConversationState::from_history(ConversationId("conv-1".into()), swapped),
        Err(ConversationError::InvalidSequence(_))
    ));
    // Duplicate: the last entry appears twice.
    let mut dup: Vec<_> = live.history().to_vec();
    dup.push(dup[1].clone());
    assert!(matches!(
        ConversationState::from_history(ConversationId("conv-1".into()), dup),
        Err(ConversationError::InvalidSequence(_))
    ));
}

/// A duplicate turn id across two entries must not silently collapse into
/// one history (Slice 6.5 replay validation).
#[test]
fn from_history_rejects_duplicate_turn_ids() {
    let live = build_two_turn_history();
    let mut entries: Vec<HistoryEntry> = live.history().to_vec();
    // Re-sequence the copied entry so only the id duplicates.
    let twin = HistoryEntry {
        sequence: TurnSequence(9),
        snapshot: entries[0].snapshot.clone(),
    };
    entries.push(twin);
    assert!(matches!(
        ConversationState::from_history(ConversationId("conv-1".into()), entries),
        Err(ConversationError::InvalidSequence(_))
    ));
}

/// A snapshot whose blocks fail pairing validation is rejected. The
/// corrupt snapshot is fabricated through serde — exactly the path
/// corrupt external data would take.
#[test]
fn from_history_rejects_unpaired_tool_results() {
    let live = build_two_turn_history();
    let blocks_json = r#"[
        {"id": {"turn_id": "bad", "sequence": 0}, "sequence": 0,
         "content": {"shape": "tool_result",
                     "value": {"call_id": "ghost", "status": "Succeeded",
                               "output": {"content": {}, "truncation": "none",
                                          "meta": null, "artifact": null}}},
         "meta": {}}
    ]"#;
    let blocks: OrderedBlocks = serde_json::from_str(blocks_json).unwrap();
    let corrupt = HistoryEntry {
        sequence: TurnSequence(9),
        snapshot: TurnSnapshot {
            turn_id: TurnId::new("bad"),
            blocks,
            source_version: ContextVersion(1),
            sealed: true,
        },
    };
    let mut entries: Vec<_> = live.history().to_vec();
    entries.push(corrupt);
    assert!(matches!(
        ConversationState::from_history(ConversationId("conv-1".into()), entries),
        Err(ConversationError::InvalidSequence(_))
    ));
}

/// A valid replay rebuilds block-for-block identical merged frames,
/// continues sequence assignment at max+1, and resets the session
/// version.
#[test]
fn replayed_conversation_matches_live_and_continues() {
    let mut live = build_two_turn_history();
    let entries: Vec<HistoryEntry> = live.history().to_vec();
    let mut replayed =
        ConversationState::from_history(ConversationId("conv-1".into()), entries.clone()).unwrap();
    assert_eq!(replayed.history_len(), 2);
    assert_eq!(replayed.version().0, 0); // replay is a fresh load

    // Both continue with the same new turn; the merged frames agree.
    for state in [&mut live, &mut replayed] {
        state.begin_turn(TurnId::new("t3")).unwrap();
        state
            .active_turn_mut()
            .unwrap()
            .append_input(TextPayload::new("in-t3"), "user")
            .unwrap();
    }
    let f_live = live.frame(RoundId(0)).unwrap();
    let f_replayed = replayed.frame(RoundId(0)).unwrap();
    assert_eq!(f_live.frame_id, f_replayed.frame_id);
    assert_eq!(
        serde_json::to_string(&f_live.model_context.blocks).unwrap(),
        serde_json::to_string(&f_replayed.model_context.blocks).unwrap()
    );

    // Sequence assignment continues at max+1 after replay.
    replayed
        .seal_turn(TurnId::new("t3"), SealedResult::Completed)
        .unwrap();
    let entry = replayed.commit(TurnId::new("t3")).unwrap();
    assert_eq!(entry.sequence, TurnSequence(2));
    assert_eq!(replayed.history_len(), 3);
}

/// Empty replay is a valid fresh session.
#[test]
fn from_history_accepts_empty_history() {
    let mut replayed =
        ConversationState::from_history(ConversationId("conv-1".into()), vec![]).unwrap();
    assert_eq!(replayed.history_len(), 0);
    assert_eq!(replayed.version().0, 0);
    commit_completed(&mut replayed, "t1");
    assert_eq!(replayed.history()[0].sequence, TurnSequence(0));
}

// ---- paused-state persistence ----------------------------------------------

/// A paused session serializes with its active turn OPEN (snapshot
/// `sealed: false`) and the Paused stamp; the round-trip restores exactly
/// that resumable shape.
#[test]
fn paused_state_round_trip_preserves_open_active_and_stamp() {
    let mut c = conv();
    c.begin_turn(TurnId::new("t1")).unwrap();
    c.active_turn_mut()
        .unwrap()
        .append_input(TextPayload::new("hi"), "user")
        .unwrap();
    c.seal_turn(TurnId::new("t1"), SealedResult::Paused)
        .unwrap();
    assert!(!c.active_turn().unwrap().is_sealed());

    let json = serde_json::to_string(&c).expect("serialize");
    let mut restored: ConversationState = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored.sealed_result(), Some(SealedResult::Paused));
    let active = restored.active_turn().expect("active preserved");
    assert!(!active.is_sealed(), "paused active must reload open");
    assert_eq!(active.turn_id(), TurnId::new("t1"));
    assert_eq!(active.blocks().len(), 1);
    // The restored pause still rejects commit.
    assert!(matches!(
        restored.commit(TurnId::new("t1")),
        Err(ConversationError::TurnPaused(_))
    ));
}

// ---- shared history, independent executions (Slice 6.5 acceptance) ----------

/// Two independent executions replay the same read-only history and each
/// begin their own turn: separate active slots, no fact pollution across
/// the branch, and source blocks keep their identity (no renumbering).
#[test]
fn two_executions_share_history_without_polluting_each_other() {
    let history: Vec<HistoryEntry> = build_two_turn_history().history().to_vec();

    let mut branch_a =
        ConversationState::from_history(ConversationId("conv-1".into()), history.clone()).unwrap();
    let mut branch_b =
        ConversationState::from_history(ConversationId("conv-1".into()), history).unwrap();

    branch_a.begin_turn(TurnId::new("branch-a")).unwrap();
    branch_b.begin_turn(TurnId::new("branch-b")).unwrap();
    branch_a
        .active_turn_mut()
        .unwrap()
        .append_input(TextPayload::new("only a sees this"), "user")
        .unwrap();
    branch_b
        .active_turn_mut()
        .unwrap()
        .append_input(TextPayload::new("only b sees this"), "user")
        .unwrap();

    let frames_a = branch_a.frame(RoundId(0)).unwrap();
    let frames_b = branch_b.frame(RoundId(0)).unwrap();
    let texts_a = serde_json::to_string(&frames_a.model_context.blocks).unwrap();
    let texts_b = serde_json::to_string(&frames_b.model_context.blocks).unwrap();
    assert!(texts_a.contains("only a sees this") && !texts_a.contains("only b sees this"));
    assert!(texts_b.contains("only b sees this") && !texts_b.contains("only a sees this"));

    // Source facts keep their identity: the first two blocks of each
    // merged view are the shared history's own BlockIds, unrenumbered.
    let shared_a: Vec<String> = frames_a.model_context.blocks[..2]
        .iter()
        .map(|b| b.id.turn_id.0.clone())
        .collect();
    let shared_b: Vec<String> = frames_b.model_context.blocks[..2]
        .iter()
        .map(|b| b.id.turn_id.0.clone())
        .collect();
    assert_eq!(shared_a, vec!["t1".to_string(), "t2".to_string()]);
    assert_eq!(shared_b, shared_a);
}
