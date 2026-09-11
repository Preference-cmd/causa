//! Minimal consumer of `causa_runtime::session`: the `new → submit → wait`
//! closed loop, plus the two guarantees the accept contract is built on
//! (per-work identity and local request-key dedup). This target pins the
//! signatures and the closed loop against the real driver; the fuller
//! acceptance suite lives in the sibling `session_*` targets.

mod common;

use std::sync::Arc;
use std::time::Duration;

use causa_kernel::{ContentPart, ConversationId, TextPayload};
use causa_runtime::{
    ConversationState, FinishedKind, Session, SessionConfig, SubmitRequest, TurnRunOptions,
    WaitEnd, WorkState,
};
use common::{RecordingGateway, endturn_output, runner_with};

#[tokio::test]
async fn new_submit_wait_closes_the_loop() {
    let gateway = RecordingGateway::scripted(vec![Ok(endturn_output("done"))]);
    let runner = Arc::new(runner_with(gateway.clone(), vec![]));
    let state = ConversationState::new(ConversationId("c1".into()));

    // `new` is an idle construction: no model call yet.
    let session = Session::new(
        state,
        runner,
        TurnRunOptions::default(),
        SessionConfig::default(),
    )
    .expect("an idle state is accepted");
    assert!(gateway.recorded().is_empty(), "new must not call the model");

    let handle = session.handle();
    assert_eq!(handle.id().0, "c1");

    let parts = vec![ContentPart::Text(TextPayload::new("hello"))];
    let receipt = handle
        .submit(SubmitRequest {
            request_key: "k1".into(),
            parts: parts.clone(),
        })
        .await
        .expect("idle session accepts the work");

    let outcome = handle
        .wait(&receipt.work, Duration::from_secs(5))
        .await
        .expect("the accepted work is observable");
    assert_eq!(outcome.end, WaitEnd::ReachedState);
    assert_eq!(outcome.observation.state, WorkState::Finished);
    match outcome.observation.finished {
        Some(FinishedKind::Completed { final_output }) => {
            assert_eq!(final_output.response.text.0, "done");
        }
        other => panic!("expected a completed work, got {other:?}"),
    }

    // The same key + parts resolves to the original receipt without a second
    // model call; the work identity is stable across the retry.
    let retried = handle
        .submit(SubmitRequest {
            request_key: "k1".into(),
            parts,
        })
        .await
        .expect("request-key dedup returns the original receipt");
    assert_eq!(retried, receipt);
    assert_eq!(
        gateway.recorded().len(),
        1,
        "the deduped submit must not run the model again"
    );

    // A finished work stays observable after the session goes idle.
    let observed = handle
        .observe(&receipt.work)
        .await
        .expect("the finished work stays readable");
    assert_eq!(observed.state, WorkState::Finished);
    assert_eq!(observed.revision, outcome.observation.revision);
}
