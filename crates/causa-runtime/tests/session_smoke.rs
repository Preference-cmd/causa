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
    CancelOutcome, ConversationState, FinishedKind, Session, SessionConfig, SessionError,
    SubmitRequest, TurnInterruption, TurnRunOptions, WaitEnd, WorkState,
};
use common::{
    EchoTool, PausingInteraction, RecordingGateway, approve, awaiting_echo, endturn_output,
    runner_with, tooluse_output,
};

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
        .expect("the finished work stays readable");
    assert_eq!(observed.state, WorkState::Finished);
    assert_eq!(observed.revision, outcome.observation.revision);
}

/// The smallest consumer of the resume / cancel / shutdown surface: a work
/// pauses, resumes to completion, a second work is cancelled while paused, and
/// the session shuts down.
#[tokio::test]
async fn paused_work_resumes_cancels_and_shuts_down() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("approval?", "echo", serde_json::json!({}))),
        Ok(endturn_output("approved")),
        Ok(tooluse_output(
            "approval again?",
            "echo",
            serde_json::json!({}),
        )),
    ]);
    let options = TurnRunOptions {
        interaction: Arc::new(PausingInteraction),
        ..Default::default()
    };
    let session = Session::new(
        ConversationState::new(ConversationId("smoke".into())),
        Arc::new(runner_with(gateway, vec![Arc::new(EchoTool)])),
        options,
        SessionConfig::default(),
    )
    .expect("an empty state is an idle session base");
    let handle = session.handle();

    let submitted = handle
        .submit(SubmitRequest {
            request_key: "s1".into(),
            parts: vec![ContentPart::Text(TextPayload::new("go"))],
        })
        .expect("an idle session accepts the work");
    let paused = handle
        .wait(&submitted.work, Duration::from_secs(5))
        .await
        .expect("the accepted work is observable");
    assert_eq!(paused.observation.state, WorkState::Paused);

    // The paused turn's unanswered call is the model-emitted one for round 0,
    // position 0; the kernel mints its call id from exactly those. Approving it
    // covers the awaiting batch exactly once.
    let awaiting = vec![awaiting_echo()];
    let resumed_receipt = handle
        .resume(
            &submitted.work,
            paused.observation.revision,
            "r1".into(),
            approve(awaiting),
        )
        .expect("the paused revision and an approving decision resume");
    assert_eq!(resumed_receipt.work, submitted.work);

    let resumed = handle
        .wait(&resumed_receipt.work, Duration::from_secs(5))
        .await
        .expect("the resumed work is observable");
    assert_eq!(resumed.observation.state, WorkState::Finished);
    match resumed.observation.finished {
        Some(FinishedKind::Completed { final_output }) => {
            assert_eq!(final_output.response.text.0, "approved");
        }
        other => panic!("expected a completed resume, got {other:?}"),
    }

    // A second work pauses too; cancelling it terminates it in place and retains
    // its paused continuation.
    let second = handle
        .submit(SubmitRequest {
            request_key: "s2".into(),
            parts: vec![ContentPart::Text(TextPayload::new("again"))],
        })
        .expect("the completed first work leaves the session idle");
    let second_paused = handle
        .wait(&second.work, Duration::from_secs(5))
        .await
        .expect("the second work is observable");
    assert_eq!(second_paused.observation.state, WorkState::Paused);

    let cancelled = handle
        .cancel(&second.work, "c1".into())
        .expect("a paused work can be cancelled");
    assert_eq!(cancelled.outcome, CancelOutcome::Stopped);
    let stopped = handle
        .wait(&second.work, Duration::from_secs(5))
        .await
        .expect("the cancelled work is observable");
    assert_eq!(stopped.observation.state, WorkState::Finished);
    match stopped.observation.finished {
        Some(FinishedKind::Interrupted {
            cause,
            continuation,
            ..
        }) => {
            assert_eq!(cause, TurnInterruption::ExplicitCancellation);
            assert!(
                continuation.is_some(),
                "a cancelled paused work retains its continuation"
            );
        }
        other => panic!("expected a cancelled work, got {other:?}"),
    }

    // Shutdown stops acceptance and winds down; retained results stay readable.
    session.shutdown().await;
    match handle.submit(SubmitRequest {
        request_key: "s3".into(),
        parts: vec![ContentPart::Text(TextPayload::new("later"))],
    }) {
        Err(SessionError::Closed) => {}
        other => panic!("expected Closed after shutdown, got {other:?}"),
    }
    assert_eq!(
        handle.observe(&submitted.work).unwrap().state,
        WorkState::Finished,
        "results remain readable after shutdown"
    );
}
