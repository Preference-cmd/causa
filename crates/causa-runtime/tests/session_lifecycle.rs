//! Session observation and owner-drop lifecycle.

mod common;

use std::time::Duration;

use causa_kernel::{ContentPart, TextPayload};
use causa_runtime::{FinishedKind, TurnInterruption, WaitEnd, WorkState};
use common::{GatedGateway, idle_session};

fn session_req(request_key: &str, text: &str) -> causa_runtime::SubmitRequest {
    causa_runtime::SubmitRequest {
        request_key: request_key.into(),
        parts: vec![ContentPart::Text(TextPayload::new(text))],
    }
}

#[tokio::test]
async fn timeout_is_observational_and_owner_drop_interrupts_work() {
    let gateway = GatedGateway::new("held", true);
    let session = idle_session("lifecycle", gateway.clone());
    let first = session.handle();
    let second = first.clone();
    let receipt = first.submit(session_req("work", "hold")).unwrap();
    gateway.wait_entered().await;

    let timeout = first
        .wait(&receipt.work, Duration::from_millis(20))
        .await
        .unwrap();
    assert_eq!(timeout.end, WaitEnd::TimedOut);
    assert_eq!(timeout.observation.state, WorkState::Running);
    assert_eq!(
        second.observe(&receipt.work).unwrap().revision,
        timeout.observation.revision
    );

    drop(session);
    let done = second
        .wait(&receipt.work, Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(done.end, WaitEnd::ReachedState);
    assert_eq!(done.observation.state, WorkState::Finished);
    match done.observation.finished {
        Some(FinishedKind::Interrupted { cause, facts, .. }) => {
            assert_eq!(cause, TurnInterruption::ExplicitCancellation);
            assert_eq!(facts.turn_id, receipt.work.turn_id);
            assert!(!facts.blocks.as_slice().is_empty());
        }
        other => panic!("expected interrupted facts, got {other:?}"),
    }
}
