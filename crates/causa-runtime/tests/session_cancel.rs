//! `SessionHandle::cancel` — signalling a running or just-accepted work,
//! stopping a paused work in place, leaving a terminal work's result alone,
//! and the request-key idempotency that makes a cancel retry safe.
//!
//! A paused work is `Stopped` with its committed facts and its continuation
//! retained; a running work is signalled and ends `Interrupted`; a terminal
//! work is `AlreadyTerminal` and its result is not rewritten; a same-key retry
//! resolves to the identical receipt; a foreign ref is `NotFound`; and a
//! completion that races the cancel keeps its own result. Shared fixtures come
//! from `tests/common`.

mod common;

use std::time::Duration;

use causa_kernel::{ConversationId, TurnId};
use causa_runtime::{
    CancelOutcome, FinishedKind, SessionError, TurnInterruption, WaitEnd, WorkRef, WorkState,
};
use common::{
    GatedGateway, RecordingGateway, endturn_output, idle_session, paused_work, session_req,
    tooluse_output,
};

// ---- helpers ----------------------------------------------------------------

// ---- the cancel outcome per state ------------------------------------------

#[tokio::test]
async fn cancel_of_a_paused_work_stops_it() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("approval?", "echo", serde_json::json!({}))),
        Ok(endturn_output("never reached")),
    ]);
    let (_session, handle, work) = paused_work("cancel-paused", gateway.clone()).await;
    let calls_before = gateway.recorded().len();

    let receipt = handle
        .cancel(&work, "stop".into())
        .expect("the paused work is cancellable");
    assert_eq!(receipt.work, work, "the receipt names the cancelled work");
    assert_eq!(receipt.outcome, CancelOutcome::Stopped);

    let observed = handle.observe(&work).expect("the work stays retained");
    assert_eq!(observed.state, WorkState::Finished);
    match &observed.finished {
        Some(FinishedKind::Interrupted {
            cause,
            facts,
            continuation,
        }) => {
            assert_eq!(
                *cause,
                TurnInterruption::ExplicitCancellation,
                "a cancelled work carries the explicit-cancellation cause"
            );
            assert_eq!(
                facts.turn_id, work.turn_id,
                "the retained facts are the cancelled work's own turn"
            );
            assert!(
                !facts.blocks.as_slice().is_empty(),
                "the paused work's real committed blocks are retained, not dropped"
            );
            assert!(
                continuation.is_some(),
                "the paused continuation is retained for inspection"
            );
        }
        other => panic!("expected an interrupted work, got {other:?}"),
    }
    assert!(observed.fault.is_none(), "a cancellation is not a fault");
    assert_eq!(
        gateway.recorded().len(),
        calls_before,
        "cancel dropped nothing and executed nothing — no second model call"
    );
}

#[tokio::test]
async fn cancelled_turn_id_is_not_reused() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("approval?", "echo", serde_json::json!({}))),
        Ok(endturn_output("second")),
    ]);
    let (_session, handle, work) = paused_work("cancel-reuse", gateway.clone()).await;

    let cancelled = handle
        .cancel(&work, "k1".into())
        .expect("the paused work is cancellable");
    assert_eq!(cancelled.outcome, CancelOutcome::Stopped);

    // The stopped work left the slot idle, so the next submit is accepted.
    let next = handle
        .submit(session_req("s2", "two"))
        .expect("a cancelled work frees the conversation");
    assert_ne!(
        next.work.turn_id, work.turn_id,
        "a cancelled TurnId is never handed to a later work"
    );
    assert_eq!(next.work.conversation_id, work.conversation_id);

    let done = handle
        .wait(&next.work, Duration::from_secs(5))
        .await
        .expect("the next work is observable");
    assert_eq!(done.observation.state, WorkState::Finished);
    match done.observation.finished {
        Some(FinishedKind::Completed { final_output }) => {
            assert_eq!(final_output.response.text.0, "second");
        }
        other => panic!("expected the next work to complete, got {other:?}"),
    }

    // The cancelled ref stays observable across the later work.
    let still = handle.observe(&work).expect("the cancelled ref stays");
    assert_eq!(still.state, WorkState::Finished);
    match &still.finished {
        Some(FinishedKind::Interrupted { cause, .. }) => {
            assert_eq!(*cause, TurnInterruption::ExplicitCancellation);
        }
        other => panic!("expected the retained cancellation, got {other:?}"),
    }
}

#[tokio::test]
async fn cancel_of_a_running_work_signals_then_interrupts() {
    let gateway = GatedGateway::new("unused", true);
    let session = idle_session("cancel-running", gateway.clone());
    let handle = session.handle();

    let receipt = handle
        .submit(session_req("r1", "hold"))
        .expect("an idle session accepts the work");
    gateway.wait_entered().await;
    assert_eq!(
        handle.observe(&receipt.work).unwrap().state,
        WorkState::Running,
        "the work is parked inside the model call"
    );

    let signalled = handle
        .cancel(&receipt.work, "c1".into())
        .expect("the running work is cancellable");
    assert_eq!(signalled.work, receipt.work);
    assert_eq!(
        signalled.outcome,
        CancelOutcome::Signalled,
        "a running work is signalled, not stopped in place"
    );

    gateway.release();
    let finished = handle
        .wait(&receipt.work, Duration::from_secs(5))
        .await
        .expect("the signalled work is observable");
    assert_eq!(finished.end, WaitEnd::ReachedState);
    assert_eq!(finished.observation.state, WorkState::Finished);
    match &finished.observation.finished {
        Some(FinishedKind::Interrupted {
            cause,
            continuation,
            ..
        }) => {
            assert_eq!(
                *cause,
                TurnInterruption::ExplicitCancellation,
                "the signalled token ends the turn as an explicit cancellation"
            );
            assert!(
                continuation.is_none(),
                "a runner-produced interruption retains no continuation"
            );
        }
        other => panic!("expected the signalled work to end interrupted, got {other:?}"),
    }

    // A same-key retry resolves to the identical receipt — no re-signal.
    let again = handle
        .cancel(&receipt.work, "c1".into())
        .expect("the same key stays valid");
    assert_eq!(
        again, signalled,
        "a same-key retry returns the original receipt"
    );

    // A fresh key on the terminal work reports it was already terminal.
    let fresh = handle
        .cancel(&receipt.work, "c2".into())
        .expect("a fresh key on the terminal work is accepted");
    assert_eq!(fresh.outcome, CancelOutcome::AlreadyTerminal);
    assert_eq!(gateway.calls(), 1, "the cancel ran no second model call");
}

#[tokio::test]
async fn cancel_of_a_terminal_work_keeps_its_result() {
    let gateway = RecordingGateway::scripted(vec![Ok(endturn_output("done"))]);
    let session = idle_session("cancel-terminal", gateway.clone());
    let handle = session.handle();

    let receipt = handle
        .submit(session_req("t1", "go"))
        .expect("an idle session accepts the work");
    let finished = handle
        .wait(&receipt.work, Duration::from_secs(5))
        .await
        .expect("the accepted work is observable");
    assert_eq!(finished.observation.state, WorkState::Finished);
    let before = match &finished.observation.finished {
        Some(FinishedKind::Completed { final_output }) => final_output.response.text.0.clone(),
        other => panic!("expected a completed work, got {other:?}"),
    };

    let cancel = handle
        .cancel(&receipt.work, "late".into())
        .expect("a terminal work is still cancellable");
    assert_eq!(cancel.outcome, CancelOutcome::AlreadyTerminal);

    let after = handle.observe(&receipt.work).expect("still retained");
    assert_eq!(after.state, WorkState::Finished);
    match &after.finished {
        Some(FinishedKind::Completed { final_output }) => assert_eq!(
            final_output.response.text.0, before,
            "a terminal cancel must not rewrite the result"
        ),
        other => panic!("expected the completion to be untouched, got {other:?}"),
    }
    assert_eq!(
        gateway.recorded().len(),
        1,
        "a terminal cancel runs no model call"
    );
}

// ---- ref validation and idempotency ----------------------------------------

#[tokio::test]
async fn cancel_foreign_ref_is_not_found() {
    let gateway = RecordingGateway::scripted(vec![Ok(endturn_output("done"))]);
    let session = idle_session("cancel-foreign", gateway.clone());
    let handle = session.handle();

    let foreign = WorkRef {
        conversation_id: ConversationId("other".into()),
        turn_id: TurnId::new("foreign-work"),
    };
    match handle.cancel(&foreign, "x".into()) {
        Err(SessionError::NotFound(ref work)) => {
            assert_eq!(*work, foreign, "the rejection names the foreign ref");
        }
        other => panic!("expected NotFound for a foreign ref, got {other:?}"),
    }
}

#[tokio::test]
async fn cancel_twice_same_key_is_stable() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("approval?", "echo", serde_json::json!({}))),
        Ok(endturn_output("never reached")),
    ]);
    let (_session, handle, work) = paused_work("cancel-idem", gateway.clone()).await;

    let first = handle
        .cancel(&work, "same".into())
        .expect("the paused work is cancellable");
    assert_eq!(first.outcome, CancelOutcome::Stopped);
    let after_first = handle.observe(&work).expect("the work is retained");
    assert_eq!(after_first.state, WorkState::Finished);

    let second = handle
        .cancel(&work, "same".into())
        .expect("the retry resolves to the original receipt");
    assert_eq!(second, first, "a same-key retry is the identical receipt");

    let after_second = handle.observe(&work).expect("the work is retained");
    assert_eq!(after_second.state, WorkState::Finished);
    assert_eq!(
        after_second.revision, after_first.revision,
        "the work was finished exactly once — the retry republished nothing"
    );
    assert_eq!(
        gateway.recorded().len(),
        1,
        "neither cancel executed a model call"
    );
}

#[tokio::test]
async fn cancel_races_a_completion_keeps_the_real_result() {
    // The model call parks inside `invoke` — past the driver's pre-round stop
    // check — and returns its end-turn output once released, even after the
    // cancel's token fired, so the completion wins the race.
    let gateway = GatedGateway::new("quick", false);
    let session = idle_session("cancel-race", gateway.clone());
    let handle = session.handle();

    let receipt = handle
        .submit(session_req("q1", "fast"))
        .expect("an idle session accepts the work");
    gateway.wait_entered().await;

    let cancel = handle
        .cancel(&receipt.work, "race".into())
        .expect("the running work is cancellable");
    assert!(
        matches!(
            cancel.outcome,
            CancelOutcome::Signalled | CancelOutcome::AlreadyTerminal
        ),
        "a cancel racing a completion is Signalled or AlreadyTerminal, got {:?}",
        cancel.outcome
    );

    gateway.release();
    let finished = handle
        .wait(&receipt.work, Duration::from_secs(5))
        .await
        .expect("the raced work is observable");
    assert_eq!(finished.end, WaitEnd::ReachedState);
    assert_eq!(finished.observation.state, WorkState::Finished);
    match &finished.observation.finished {
        Some(FinishedKind::Completed { final_output }) => {
            assert_eq!(
                final_output.response.text.0, "quick",
                "a completion that wins the race keeps the model's real output"
            );
        }
        other => panic!("a completion that wins the race is never rewritten, got {other:?}"),
    }
    assert!(
        finished.observation.fault.is_none(),
        "a raced completion is not a fault"
    );
}

#[tokio::test]
async fn cancel_immediately_after_submit_signals_the_accepted_work() {
    let gateway = GatedGateway::new("unused", true);
    let session = idle_session("cancel-accepted", gateway.clone());
    let handle = session.handle();

    let receipt = handle
        .submit(session_req("r1", "hold"))
        .expect("an idle session accepts the work");
    // No `wait_entered`: `submit` switches the slot to `Running`
    // synchronously under the registry lock, so the cancel deterministically
    // signals even when the worker has not marked itself `Running` yet. An
    // accepted-but-not-started work is signalled, never stopped in place.
    let signalled = handle
        .cancel(&receipt.work, "c1".into())
        .expect("the accepted work is cancellable");
    assert_eq!(signalled.work, receipt.work);
    assert_eq!(
        signalled.outcome,
        CancelOutcome::Signalled,
        "an accepted work is signalled like a running one"
    );

    gateway.release();
    let finished = handle
        .wait(&receipt.work, Duration::from_secs(5))
        .await
        .expect("the signalled work is observable");
    assert_eq!(finished.end, WaitEnd::ReachedState);
    assert_eq!(finished.observation.state, WorkState::Finished);
    match &finished.observation.finished {
        Some(FinishedKind::Interrupted { cause, .. }) => {
            assert_eq!(
                *cause,
                TurnInterruption::ExplicitCancellation,
                "the signalled token ends the accepted work as an explicit cancellation"
            );
        }
        other => panic!("expected the accepted work to end interrupted, got {other:?}"),
    }
    // The cancel may win before the first model call (no call at all) or
    // while parked inside it (exactly one call) — either way it must never
    // cause a second call.
    assert!(gateway.calls() <= 1, "the cancel runs no second model call");
}

#[tokio::test]
async fn cancel_same_key_for_another_work_conflicts_before_not_found() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("approval?", "echo", serde_json::json!({}))),
        Ok(endturn_output("never reached")),
    ]);
    let (_session, handle, work) = paused_work("cancel-key-scope", gateway.clone()).await;

    let first = handle
        .cancel(&work, "c1".into())
        .expect("the paused work is cancellable");
    assert_eq!(first.outcome, CancelOutcome::Stopped);

    // The key is taken even though the named work was never accepted: dedup
    // resolves before the ref lookup, so a reused key can never be mistaken
    // for a fresh cancel on another work.
    let unknown = WorkRef {
        conversation_id: ConversationId("cancel-key-scope".into()),
        turn_id: TurnId::new("never-accepted"),
    };
    match handle.cancel(&unknown, "c1".into()) {
        Err(SessionError::Conflict) => {}
        other => panic!("expected Conflict for a reused cancel key, got {other:?}"),
    }
}
