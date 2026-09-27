//! Shutdown closes admission and retains terminal observations.

mod common;

use std::time::Duration;

use causa_kernel::{ContentPart, TextPayload};
use causa_runtime::{FinishedKind, SessionError, WorkState};
use common::{GatedGateway, idle_session};

fn session_req(request_key: &str, text: &str) -> causa_runtime::SubmitRequest {
    causa_runtime::SubmitRequest {
        request_key: request_key.into(),
        parts: vec![ContentPart::Text(TextPayload::new(text))],
    }
}

#[tokio::test]
async fn shutdown_cancels_running_work_and_keeps_its_result_readable() {
    let gateway = GatedGateway::new("held", true);
    let session = idle_session("shutdown", gateway.clone());
    let handle = session.handle();
    let receipt = handle.submit(session_req("first", "hold")).unwrap();
    gateway.wait_entered().await;

    session.shutdown().await;
    assert!(matches!(
        handle.submit(session_req("later", "no")),
        Err(SessionError::Closed)
    ));
    assert_eq!(
        handle.submit(session_req("first", "hold")).unwrap(),
        receipt,
        "an accepted submit key remains replayable after close"
    );
    let observed = handle.observe(&receipt.work).unwrap();
    assert_eq!(observed.state, WorkState::Finished);
    assert!(matches!(
        observed.finished,
        Some(FinishedKind::Interrupted { .. })
    ));
    assert_eq!(
        handle
            .wait(&receipt.work, Duration::from_secs(1))
            .await
            .unwrap()
            .observation
            .state,
        WorkState::Finished
    );
}

#[tokio::test]
async fn idle_shutdown_is_immediate_and_idempotent() {
    let session = common::idle_session("idle-shutdown", common::RecordingGateway::scripted(vec![]));
    session.shutdown().await;
    session.shutdown().await;
    assert!(matches!(
        session.handle().submit(session_req("after", "no")),
        Err(SessionError::Closed)
    ));
}
