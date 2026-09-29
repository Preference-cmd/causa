//! Cancellation is idempotent and leaves an interrupted work observable.

mod common;

use std::time::Duration;

use causa_kernel::{ContentPart, ConversationId, TextPayload, TurnId};
use causa_runtime::{
    CancelOutcome, FinishedKind, SessionError, SubmitRequest, TurnInterruption, WorkRef, WorkState,
};
use common::{GatedGateway, idle_session};

fn session_req(request_key: &str, text: &str) -> SubmitRequest {
    SubmitRequest {
        request_key: request_key.into(),
        parts: vec![ContentPart::Text(TextPayload::new(text))],
    }
}

#[tokio::test]
async fn cancel_signals_running_work_and_replays_its_receipt() {
    let gateway = GatedGateway::new("held", true);
    let session = idle_session("cancel-running", gateway.clone());
    let handle = session.handle();
    let submitted = handle.submit(session_req("submit", "work")).unwrap();
    gateway.wait_entered().await;

    let first = handle.cancel(&submitted.work, "cancel".into()).unwrap();
    assert_eq!(first.outcome, CancelOutcome::Signalled);
    assert_eq!(
        handle.cancel(&submitted.work, "cancel".into()).unwrap(),
        first
    );
    let other = WorkRef {
        conversation_id: ConversationId("other-session".into()),
        turn_id: TurnId::new("other-work"),
    };
    assert_eq!(
        handle.cancel(&other, "cancel".into()).unwrap_err(),
        SessionError::Conflict,
        "a reused cancel key cannot target a different work"
    );

    let waited = handle
        .wait(&submitted.work, Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(waited.observation.state, WorkState::Finished);
    match waited.observation.finished {
        Some(FinishedKind::Interrupted { cause, facts, .. }) => {
            assert_eq!(cause, TurnInterruption::ExplicitCancellation);
            assert_eq!(facts.turn_id(), submitted.work.turn_id);
            assert!(!facts.blocks().is_empty());
        }
        other => panic!("expected retained interrupted facts, got {other:?}"),
    }

    let terminal = handle
        .cancel(&submitted.work, "after-finish".into())
        .unwrap();
    assert_eq!(terminal.outcome, CancelOutcome::AlreadyTerminal);

    let next = handle.submit(session_req("next", "another turn")).unwrap();
    assert_ne!(next.work.turn_id, submitted.work.turn_id);
    gateway.wait_entered().await;
    gateway.release();
    assert_eq!(
        handle
            .wait(&next.work, Duration::from_secs(5))
            .await
            .unwrap()
            .observation
            .state,
        WorkState::Finished
    );
}
