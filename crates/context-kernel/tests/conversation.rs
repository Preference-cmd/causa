//! ConversationState Phase A acceptance — aggregate facts and order.
//! No runner involved: the test plays the driver's stamping role via
//! `seal_turn` (the same seam the runner uses from Phase C on).

use reimagine_context_kernel::{
    ConversationError, ConversationId, ConversationState, FrameScope, RoundId, SealedResult,
    TextPayload, TurnId,
};

fn conv() -> ConversationState {
    ConversationState::new(ConversationId("conv-1".into()))
}

/// Canonical driver flow: begin, admit facts through the door, seal with the
/// outcome stamp, commit into history.
fn commit_completed(state: &mut ConversationState, turn_id: &str) {
    state.begin_turn(TurnId::new(turn_id)).unwrap();
    state
        .active_turn_mut()
        .unwrap()
        .append_input(TextPayload::new("hi"), "user")
        .unwrap();
    state
        .seal_turn(TurnId::new(turn_id), SealedResult::Completed)
        .unwrap();
    state.commit(TurnId::new(turn_id)).unwrap();
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
    let snap = c.commit(TurnId::new("t1")).unwrap();
    assert_eq!(snap.turn_sequence.0, 0);
    assert_eq!(snap.turn_id.0, "t1");
    // Repeated commit: the active slot is empty, so rejection lands on
    // UnknownTurn — rejection, not idempotence.
    assert!(matches!(
        c.commit(TurnId::new("t1")),
        Err(ConversationError::UnknownTurn(_))
    ));
    assert_eq!(c.snapshot_count(), 1);
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
    assert_eq!(c.snapshot_count(), 1);
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
    assert_eq!(a.snapshot_count(), 1);
    assert_eq!(b.snapshot_count(), 0);
    assert_eq!(b.version().0, 0);
    // The same turn id lives independently in each conversation.
    commit_completed(&mut b, "shared-id");
    assert_eq!(b.snapshot_count(), 1);
    assert_eq!(b.completed_turns()[0].turn_sequence.0, 0);
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
    let snap = c.commit(TurnId::new("t1")).unwrap();
    assert_eq!(snap.blocks.as_slice().len(), 1);
    // Aborting a later turn leaves committed history byte-identical.
    c.begin_turn(TurnId::new("t2")).unwrap();
    c.seal_turn(TurnId::new("t2"), SealedResult::Interrupted)
        .unwrap();
    c.abort_turn(TurnId::new("t2")).unwrap();
    assert_eq!(c.snapshot_count(), 1);
    assert_eq!(c.completed_turns()[0].turn_id.0, "t1");
}

// ---- Phase B: lossless merged view ------------------------------------------

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
    // history blocks in TurnSequence order, then active blocks.
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
    let before_history = serde_json::to_string(c.completed_turns()).unwrap();
    let before_active = serde_json::to_string(&c.active_turn().unwrap().snapshot_blocks()).unwrap();
    let _ = c.frame(RoundId(0)).unwrap();
    assert_eq!(
        serde_json::to_string(c.completed_turns()).unwrap(),
        before_history
    );
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
    // Pure-history inspection goes through completed_turns(); the merged
    // frame needs an active turn for its scope identity.
    assert!(matches!(
        c.frame(RoundId(0)),
        Err(ConversationError::NoActiveTurn)
    ));
}
