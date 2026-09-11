//! Shutdown acceptance for the session owner.
//!
//! `Session::shutdown` stops acceptance and then waits for the running work to
//! wind down: it returns only once the worker has published its terminal state,
//! and it leaves retrievable results and paused material readable through
//! surviving handles. A fresh submit after shutdown is `Closed`; a faulted work
//! is never replayed; a repeat shutdown is harmless; an idle shutdown returns at
//! once.
//!
//! Reuses the shared `common` fixtures; every test runs offline.

mod common;

use std::sync::Arc;
use std::time::Duration;

use causa_kernel::{ContentPart, ConversationId, TextPayload};
use causa_runtime::{
    ConversationState, FinishedKind, Session, SessionConfig, SessionError, SubmitRequest,
    TurnInterruption, TurnRunOptions, WaitEnd, WorkState,
};
use common::{
    EchoTool, GatedGateway, PanickingGateway, PausingInteraction, RecordingGateway, approve,
    awaiting_echo, idle_session, runner_with, tooluse_output,
};

// ---- helpers ---------------------------------------------------------------

fn req(key: &str, text: &str) -> SubmitRequest {
    SubmitRequest {
        request_key: key.into(),
        parts: vec![ContentPart::Text(TextPayload::new(text))],
    }
}

// ---- shutdown waits for the worker to wind down ----------------------------

#[tokio::test]
async fn shutdown_winds_down_a_running_work() {
    let gateway = GatedGateway::new("gated", true);
    let session = idle_session("shutdown-wind", gateway.clone());
    let handle = session.handle();

    let receipt = handle
        .submit(req("w", "hold"))
        .expect("an idle session accepts the work");
    let work = receipt.work.clone();
    gateway.wait_entered().await;
    assert_eq!(
        handle.observe(&work).unwrap().state,
        WorkState::Running,
        "the work is parked inside the model call"
    );

    session.shutdown().await;

    // shutdown must not return before the worker published: an immediate
    // observe reads the terminal state, never a leftover `Running`.
    let observed = handle
        .observe(&work)
        .expect("a retained work is readable after shutdown");
    assert_eq!(
        observed.state,
        WorkState::Finished,
        "shutdown returns only after the worker has wound down"
    );
    match observed.finished {
        Some(FinishedKind::Interrupted { cause, .. }) => {
            assert_eq!(cause, TurnInterruption::ExplicitCancellation);
        }
        other => panic!("expected an interrupted work, got {other:?}"),
    }
}

// ---- shutdown stops acceptance but keeps retained results ------------------

#[tokio::test]
async fn after_shutdown_submit_is_closed() {
    let gateway = GatedGateway::new("gated", true);
    let session = idle_session("shutdown-closed", gateway.clone());
    let handle = session.handle();

    let receipt = handle
        .submit(req("first", "hold"))
        .expect("an idle session accepts the work");
    gateway.wait_entered().await;

    session.shutdown().await;

    // No new work is accepted once shut down.
    match handle.submit(req("second", "later")) {
        Err(SessionError::Closed) => {}
        other => panic!("expected SessionError::Closed after shutdown, got {other:?}"),
    }

    // The retained finished work is still observable.
    let retained = handle
        .observe(&receipt.work)
        .expect("a retained work stays observable after shutdown");
    assert_eq!(retained.state, WorkState::Finished);
    assert!(matches!(
        retained.finished,
        Some(FinishedKind::Interrupted { .. })
    ));
}

// ---- shutdown neither approves nor discards a pause ------------------------

#[tokio::test]
async fn shutdown_retains_paused_material() {
    let gateway = RecordingGateway::scripted(vec![Ok(tooluse_output(
        "approval?",
        "echo",
        serde_json::json!({}),
    ))]);
    let options = TurnRunOptions {
        interaction: Arc::new(PausingInteraction),
        ..Default::default()
    };
    let session = Session::new(
        ConversationState::new(ConversationId("shutdown-paused".into())),
        Arc::new(runner_with(gateway.clone(), vec![Arc::new(EchoTool)])),
        options,
        SessionConfig::default(),
    )
    .expect("an empty state is an idle session base");
    let handle = session.handle();

    let receipt = handle
        .submit(req("p", "approve me"))
        .expect("an idle session accepts the work");
    let paused = handle
        .wait(&receipt.work, Duration::from_secs(5))
        .await
        .expect("the accepted work is observable");
    assert_eq!(paused.end, WaitEnd::ReachedState);
    assert_eq!(paused.observation.state, WorkState::Paused);

    session.shutdown().await;

    // The pause is retained as it was: not approved (still `Paused`, no
    // `FinishedKind`) and not discarded (still observable at its revision).
    let retained = handle
        .observe(&receipt.work)
        .expect("a paused work stays observable after shutdown");
    assert_eq!(
        retained.state,
        WorkState::Paused,
        "shutdown approves nothing"
    );
    assert!(
        retained.finished.is_none(),
        "shutdown must not fabricate a finished kind for a pause"
    );
    assert!(
        retained.fault.is_none(),
        "a pause is not turned into a fault"
    );
    assert_eq!(
        retained.revision, paused.observation.revision,
        "shutdown leaves the paused revision untouched"
    );
    assert_eq!(
        gateway.recorded().len(),
        1,
        "shutting down a paused work runs no extra model call"
    );
}

// ---- shutdown is idempotent ------------------------------------------------

#[tokio::test]
async fn shutdown_is_idempotent() {
    let gateway = GatedGateway::new("gated", true);
    let session = idle_session("shutdown-idem", gateway.clone());
    let handle = session.handle();

    let receipt = handle
        .submit(req("w", "hold"))
        .expect("an idle session accepts the work");
    gateway.wait_entered().await;

    session.shutdown().await;
    let after_first = handle
        .observe(&receipt.work)
        .expect("retained after the first shutdown");
    assert_eq!(after_first.state, WorkState::Finished);

    session.shutdown().await;
    let after_second = handle
        .observe(&receipt.work)
        .expect("retained after the second shutdown");

    // A repeat shutdown does not rewrite or hide the retained work.
    assert_eq!(after_second.state, WorkState::Finished);
    assert_eq!(
        after_second.revision, after_first.revision,
        "a repeat shutdown must not mutate the retained work"
    );
    assert!(matches!(
        after_second.finished,
        Some(FinishedKind::Interrupted { .. })
    ));
    assert_eq!(after_second.fault, after_first.fault);
}

// ---- a fault is not replayed by shutdown -----------------------------------

#[tokio::test]
async fn faulted_is_not_replayed_by_shutdown() {
    let session = idle_session("shutdown-fault", Arc::new(PanickingGateway));
    let handle = session.handle();

    let receipt = handle
        .submit(req("w", "boom"))
        .expect("an idle session accepts the work");
    let faulted = handle
        .wait(&receipt.work, Duration::from_secs(5))
        .await
        .expect("a faulted work is observable, never a permanent Running");
    assert_eq!(faulted.end, WaitEnd::ReachedState);
    assert_eq!(faulted.observation.state, WorkState::Faulted);
    let reason = faulted
        .observation
        .fault
        .clone()
        .expect("a fault publishes its reason");

    session.shutdown().await;

    // The fault is retained as-is: not auto-replayed, and no fabricated scene.
    let retained = handle
        .observe(&receipt.work)
        .expect("a faulted work stays observable after shutdown");
    assert_eq!(
        retained.state,
        WorkState::Faulted,
        "shutdown must not replay the faulted work"
    );
    assert_eq!(retained.fault.as_deref(), Some(reason.as_str()));
    assert!(
        retained.finished.is_none(),
        "a fault must not fake a finished kind"
    );

    match handle.submit(req("next", "after")) {
        Err(SessionError::Closed) => {}
        other => panic!("expected SessionError::Closed after shutdown, got {other:?}"),
    }
}

// ---- an idle shutdown returns at once --------------------------------------

#[tokio::test]
async fn shutdown_with_no_active_work_returns_immediately() {
    let session = idle_session("shutdown-idle", RecordingGateway::scripted(vec![]));
    let handle = session.handle();

    // No worker is running, so shutdown settles without waiting on anything.
    tokio::time::timeout(Duration::from_secs(5), session.shutdown())
        .await
        .expect("an idle shutdown does not wait on a worker");

    match handle.submit(req("w", "later")) {
        Err(SessionError::Closed) => {}
        other => panic!("expected SessionError::Closed after shutdown, got {other:?}"),
    }
}

// ---- after shutdown, control ops close but retained material stays readable --

#[tokio::test]
async fn after_shutdown_resume_and_cancel_are_closed_but_paused_material_stays_readable() {
    let gateway = RecordingGateway::scripted(vec![Ok(tooluse_output(
        "approval?",
        "echo",
        serde_json::json!({}),
    ))]);
    let options = TurnRunOptions {
        interaction: Arc::new(PausingInteraction),
        ..Default::default()
    };
    let session = Session::new(
        ConversationState::new(ConversationId("shutdown-controls".into())),
        Arc::new(runner_with(gateway.clone(), vec![Arc::new(EchoTool)])),
        options,
        SessionConfig::default(),
    )
    .expect("an empty state is an idle session base");
    let handle = session.handle();

    let receipt = handle
        .submit(req("p", "approve me"))
        .expect("an idle session accepts the work");
    let paused = handle
        .wait(&receipt.work, Duration::from_secs(5))
        .await
        .expect("the accepted work is observable");
    assert_eq!(paused.observation.state, WorkState::Paused);

    session.shutdown().await;

    // Control operations stop at the closed gate without touching anything.
    match handle.resume(
        &receipt.work,
        paused.observation.revision,
        "r".into(),
        approve(vec![awaiting_echo()]),
    ) {
        Err(SessionError::Closed) => {}
        other => panic!("expected SessionError::Closed for resume after shutdown, got {other:?}"),
    }
    match handle.cancel(&receipt.work, "c".into()) {
        Err(SessionError::Closed) => {}
        other => panic!("expected SessionError::Closed for cancel after shutdown, got {other:?}"),
    }
    assert_eq!(
        gateway.recorded().len(),
        1,
        "closed control ops run no model call"
    );

    // Retained material stays readable through the surviving handle,
    // including a finite wait that still reports the pause.
    let retained = handle
        .observe(&receipt.work)
        .expect("a paused work stays observable after shutdown");
    assert_eq!(retained.state, WorkState::Paused);
    assert_eq!(retained.revision, paused.observation.revision);
    assert!(retained.finished.is_none());
    assert!(retained.fault.is_none());
    let waited = handle
        .wait(&receipt.work, Duration::from_millis(200))
        .await
        .expect("waiting a retained pause is still observable");
    assert_eq!(waited.end, WaitEnd::ReachedState);
    assert_eq!(waited.observation.state, WorkState::Paused);
}
