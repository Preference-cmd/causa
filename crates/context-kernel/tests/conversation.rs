//! ConversationState Phase A acceptance — aggregate facts and order.
//! No runner involved: the test plays the driver's stamping role via
//! `seal_turn` (the same seam the runner uses from Phase C on).

mod common;

use common::{
    DropAllCompaction, RecordingGateway, commit_sealed, ctrl, endturn_output, runner_with,
};
use std::sync::Arc;

use reimagine_context_kernel::{
    ContextVersion, ConversationError, ConversationId, ConversationState, FramePolicy, FrameScope,
    ModelInvokeErrorKind, OrderedBlocks, RoundId, SealedResult, TextPayload, TurnContext, TurnId,
    TurnOutcome, TurnResult, TurnRunOptions, TurnSequence, TurnSnapshot, WindowBudget,
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

// ---- Phase C: runner seam ----------------------------------------------------

/// Acceptance #1: both entries run the same state machine — same input
/// sequence yields the same terminal result, round count, and facts.
#[tokio::test]
async fn dual_entries_share_one_state_machine() {
    let runner = runner_with(
        RecordingGateway::repeating_last(vec![Ok(endturn_output("done"))]),
        vec![],
    );
    // Single-turn entry — same turn id as the conversation path, so the
    // accumulated facts must be byte-identical.
    let mut ctx = TurnContext::new(TurnId::new("t1"));
    ctx.append_input(TextPayload::new("hi"), "user").unwrap();
    let single: TurnOutcome = runner.run(ctx, TurnRunOptions::default(), ctrl()).await;
    // Conversation entry with empty history.
    let mut state = ConversationState::new(ConversationId("conv-1".into()));
    state.begin_turn(TurnId::new("t1")).unwrap();
    state
        .active_turn_mut()
        .unwrap()
        .append_input(TextPayload::new("hi"), "user")
        .unwrap();
    let mut conv = runner
        .run_in_conversation(state, TurnRunOptions::default(), ctrl())
        .await
        .unwrap();
    assert!(matches!(single.result, TurnResult::Completed { .. }));
    assert!(matches!(conv.result, TurnResult::Completed { .. }));
    assert_eq!(single.trace.rounds.len(), conv.trace.rounds.len());
    // Same facts accumulated in the active turn.
    assert_eq!(
        serde_json::to_string(single.context.blocks()).unwrap(),
        serde_json::to_string(conv.state.active_turn().unwrap().blocks()).unwrap()
    );
    // The conversation state comes back sealed and stamped, not yet committed.
    assert!(conv.state.active_turn().unwrap().is_sealed());
    assert_eq!(conv.state.snapshot_count(), 0);
    // The host loop completes: commit receives the turn into history.
    let snap = conv.state.commit(TurnId::new("t1")).unwrap();
    assert_eq!(snap.turn_sequence.0, 0);
    assert_eq!(conv.state.version().0, 2);
}

/// Acceptance #13: caller bugs fail fast at the entry, before the machine.
#[tokio::test]
async fn conversation_entry_rejects_missing_or_sealed_active() {
    let runner = runner_with(
        RecordingGateway::repeating_last(vec![Ok(endturn_output("x"))]),
        vec![],
    );
    let state = ConversationState::new(ConversationId("conv-1".into()));
    assert!(matches!(
        runner
            .run_in_conversation(state, TurnRunOptions::default(), ctrl())
            .await,
        Err(ConversationError::NoActiveTurn)
    ));
    let mut state = ConversationState::new(ConversationId("conv-1".into()));
    state.begin_turn(TurnId::new("t1")).unwrap();
    state
        .seal_turn(TurnId::new("t1"), SealedResult::Completed)
        .unwrap();
    assert!(matches!(
        runner
            .run_in_conversation(state, TurnRunOptions::default(), ctrl())
            .await,
        Err(ConversationError::TurnAlreadySealed)
    ));
}

/// Acceptance #16: the conversation entry is inert to `options.frame` — a
/// compacting policy that would empty a single-turn frame leaves the merged
/// frame lossless (the model still sees every block).
#[tokio::test]
async fn conversation_entry_is_inert_to_frame_policy() {
    let gateway = RecordingGateway::repeating_last(vec![Ok(endturn_output("done"))]);
    let runner = runner_with(gateway.clone(), vec![]);
    let inert = FramePolicy {
        window_budget: WindowBudget {
            model_window_limit: 100,
            compaction_trigger: 1,
        },
        compaction: Some(Arc::new(DropAllCompaction)),
        token_counter: None,
    };
    let options = TurnRunOptions {
        frame: inert,
        ..Default::default()
    };
    let mut state = ConversationState::new(ConversationId("conv-1".into()));
    state.begin_turn(TurnId::new("t1")).unwrap();
    state
        .active_turn_mut()
        .unwrap()
        .append_input(TextPayload::new("hi"), "user")
        .unwrap();
    let out = runner
        .run_in_conversation(state, options, ctrl())
        .await
        .unwrap();
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    // The model's frame contained the full history + active blocks — the
    // compacting policy never touched the merged view.
    let frames = gateway.frames();
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].model_context.blocks.len(), 1);
    // History and active facts are untouched either way.
    assert_eq!(out.state.snapshot_count(), 0);
}

/// The Interrupted flow end to end: the runner seals and stamps Interrupted,
/// commit refuses, abort discards, history stays empty.
#[tokio::test]
async fn interrupted_conversation_turn_is_stamped_and_aborted() {
    let runner = runner_with(
        RecordingGateway::scripted(vec![Err(ModelInvokeErrorKind::Permanent)]),
        vec![],
    );
    let mut state = ConversationState::new(ConversationId("conv-1".into()));
    state.begin_turn(TurnId::new("t1")).unwrap();
    state
        .active_turn_mut()
        .unwrap()
        .append_input(TextPayload::new("hi"), "user")
        .unwrap();
    let mut out = runner
        .run_in_conversation(state, TurnRunOptions::default(), ctrl())
        .await
        .unwrap();
    assert!(matches!(out.result, TurnResult::Interrupted { .. }));
    assert!(out.state.active_turn().unwrap().is_sealed());
    assert!(matches!(
        out.state.commit(TurnId::new("t1")),
        Err(ConversationError::TurnNotCompleted(_))
    ));
    out.state.abort_turn(TurnId::new("t1")).unwrap();
    assert_eq!(out.state.snapshot_count(), 0);
}

// ---- Phase D: validated replay ----------------------------------------------

/// Acceptance #14: out-of-order and duplicate turn sequences are rejected.
#[test]
fn from_snapshots_rejects_non_monotonic_sequences() {
    let live = build_two_turn_history();
    // Out of order: [seq 1, seq 0].
    let swapped: Vec<_> = live.completed_turns().iter().rev().cloned().collect();
    assert!(matches!(
        ConversationState::from_snapshots(ConversationId("conv-1".into()), swapped),
        Err(ConversationError::InvalidSequence(_))
    ));
    // Duplicate: the last snapshot appears twice.
    let mut dup: Vec<_> = live.completed_turns().to_vec();
    dup.push(dup[1].clone());
    assert!(matches!(
        ConversationState::from_snapshots(ConversationId("conv-1".into()), dup),
        Err(ConversationError::InvalidSequence(_))
    ));
}

/// Acceptance #14: a snapshot whose blocks fail pairing validation is
/// rejected. The corrupt snapshot is fabricated through serde — exactly the
/// path corrupt external data would take.
#[test]
fn from_snapshots_rejects_unpaired_tool_results() {
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
    let corrupt = TurnSnapshot {
        turn_id: TurnId::new("bad"),
        turn_sequence: TurnSequence(9),
        blocks,
        source_version: ContextVersion(1),
    };
    let mut snapshots: Vec<_> = live.completed_turns().to_vec();
    snapshots.push(corrupt);
    assert!(matches!(
        ConversationState::from_snapshots(ConversationId("conv-1".into()), snapshots),
        Err(ConversationError::InvalidSequence(_))
    ));
}

/// Acceptance #14/#15: a valid replay rebuilds block-for-block identical
/// merged frames, continues sequence assignment at max+1, and resets the
/// conversation version.
#[test]
fn replayed_conversation_matches_live_and_continues() {
    let mut live = build_two_turn_history();
    let snapshots: Vec<_> = live.completed_turns().to_vec();
    let mut replayed =
        ConversationState::from_snapshots(ConversationId("conv-1".into()), snapshots.clone())
            .unwrap();
    assert_eq!(replayed.snapshot_count(), 2);
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
    let snap = replayed.commit(TurnId::new("t3")).unwrap();
    assert_eq!(snap.turn_sequence, TurnSequence(2));
    assert_eq!(replayed.snapshot_count(), 3);
}

/// Empty replay is a valid fresh conversation.
#[test]
fn from_snapshots_accepts_empty_history() {
    let mut replayed =
        ConversationState::from_snapshots(ConversationId("conv-1".into()), vec![]).unwrap();
    assert_eq!(replayed.snapshot_count(), 0);
    assert_eq!(replayed.version().0, 0);
    commit_completed(&mut replayed, "t1");
    assert_eq!(replayed.completed_turns()[0].turn_sequence, TurnSequence(0));
}
