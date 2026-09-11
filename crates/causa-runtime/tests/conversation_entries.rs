//! Conversation-entry driver tests — the dual-entry acceptance set. The
//! fact-machine tests (commit/seal/replay) stay in the kernel; these
//! exercise the `run` / `run_in_conversation` entries.

mod common;

use common::{DropAllCompaction, RecordingGateway, ctrl, endturn_output, runner_with};
use std::sync::Arc;

use causa_kernel::{ConversationId, ModelInvokeErrorKind, TextPayload, TurnContext, TurnId};
use causa_runtime::{
    ConversationError, ConversationState, FramePolicy, SealedResult, TurnOutcome, TurnResult,
    TurnRunOptions, WindowBudget,
};

/// Both entries run the same state machine — same input sequence yields
/// the same terminal result, round count, and facts.
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
    assert_eq!(conv.state.history_len(), 0);
    // The host loop completes: commit receives the turn into history.
    let entry = conv.state.commit(TurnId::new("t1")).unwrap();
    assert_eq!(entry.sequence.0, 0);
    assert_eq!(conv.state.version().0, 2);
}

/// Caller bugs fail fast at the entry, before the machine.
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

/// The conversation entry is inert to `options.frame` — a compacting policy
/// that would empty a single-turn frame still leaves the merged frame lossless
/// (the model sees every block).
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
    assert_eq!(out.state.history_len(), 0);
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
    assert_eq!(out.state.history_len(), 0);
}
