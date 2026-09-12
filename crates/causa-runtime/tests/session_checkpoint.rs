//! Slice 8 Phase C acceptance: the material-free save loop —
//! `pause → checkpoint → restore → explicit resume` — plus the export
//! guards, the replay semantics, and the validation rejections.
//!
//! C1 — an idle or paused session exports a versioned envelope (the paused
//! phase carries the complete outcome, the state saved once inside it);
//! accepted / running works refuse with `Busy`, a faulted session refuses,
//! and a closed session still exports with `closed` recorded.
//!
//! C2 — restore reloads the work refs, revisions, allocation progress,
//! request-key tables, receipts, results, and deadlines: same-key requests
//! replay their original receipts, different arguments still conflict, the
//! next submit continues the TurnId sequence, old results stay observable,
//! and a restored work resumes under its remaining — never re-granted —
//! deadline.
//!
//! C3 — restore validates version, the assembled configuration description,
//! and material consistency before registering; every rejection hands the
//! original checkpoint back; a successful restore calls no model and no
//! tool, and a restored pause still needs an explicit resume that continues
//! the original continuation.
//!
//! C4 — offline evidence for a config mismatch, a conflict identity, an
//! expired deadline, and a closed owner handover; opening from history
//! stays a different operation from restoring a checkpoint (no dedup
//! table).

mod common;

use std::sync::Arc;
use std::time::Duration;

use causa_kernel::{ContentPart, ConversationId, ModelRef, TextPayload, TurnId};
use causa_runtime::{
    CheckpointPhase, FinishedKind, PausePoint, ResumeRequest, SESSION_CHECKPOINT_VERSION, Session,
    SessionCheckpoint, SessionConfig, SessionConfigDescription, SessionError, SubmitRequest,
    TurnInterruption, TurnResult, TurnRunOptions, WaitEnd, WorkRef, WorkState,
};
use common::{
    EchoTool, GatedGateway, RecordingGateway, approve, awaiting_echo, endturn_output, idle_session,
    pausing_session, runner_with, session_req, submit_to_pause, tooluse_output,
};
use serde_json::json;

/// A paused session plus its export: the first model call emits an `echo`
/// batch (empty arguments, so [`awaiting_echo`] matches it) that the
/// interaction pauses. The session stays owned by the caller — dropping it
/// would close the session.
async fn paused_checkpoint(
    id: &str,
    config: SessionConfig,
) -> (Session, WorkRef, u64, SessionCheckpoint) {
    let gateway =
        RecordingGateway::scripted(vec![Ok(tooluse_output("approve?", "echo", json!({})))]);
    let session = pausing_session(id, gateway, vec![Arc::new(EchoTool)], config);
    let handle = session.handle();
    let (receipt, paused) = submit_to_pause(&handle, "pause").await;
    let checkpoint = session
        .checkpoint()
        .expect("a paused session exports its envelope");
    (session, receipt.work, paused.revision, checkpoint)
}

/// Restore `checkpoint` against the same default assembly, driving the fresh
/// `gateway` afterwards.
fn restore(
    checkpoint: SessionCheckpoint,
    gateway: Arc<RecordingGateway>,
    config: SessionConfig,
) -> Session {
    SessionCheckpoint::restore(
        checkpoint,
        Arc::new(runner_with(gateway, Vec::new())),
        TurnRunOptions::default(),
        config,
    )
    .expect("the exported envelope restores against the same assembly")
}

// ---- C1: export guards --------------------------------------------------------

#[tokio::test]
async fn an_idle_session_exports_a_versioned_envelope() {
    let session = idle_session("c1-idle", RecordingGateway::scripted(vec![]));
    let checkpoint = session.checkpoint().expect("an idle session exports");

    assert_eq!(checkpoint.version, SESSION_CHECKPOINT_VERSION);
    assert_eq!(checkpoint.conversation_id, ConversationId("c1-idle".into()));
    assert!(matches!(checkpoint.phase, CheckpointPhase::Idle { .. }));
    assert!(checkpoint.works.is_empty());
    assert_eq!(checkpoint.next_turn, 0);
    assert!(checkpoint.submit_keys.is_empty());
    assert!(!checkpoint.closed);
    assert_eq!(
        checkpoint.config,
        SessionConfigDescription::of(&TurnRunOptions::default(), &SessionConfig::default()),
        "the description matches the assembled options"
    );
}

#[tokio::test]
async fn a_paused_session_exports_the_complete_outcome_once() {
    let (_, work, _, checkpoint) = paused_checkpoint("c1-paused", SessionConfig::default()).await;

    let CheckpointPhase::Paused {
        outcome,
        work: paused_ref,
    } = &checkpoint.phase
    else {
        panic!("the paused session exports the paused phase");
    };
    assert_eq!(paused_ref, &work);
    assert!(
        matches!(&outcome.result, TurnResult::Paused { .. }),
        "the outcome carries the paused result and its continuation"
    );
    // The state is saved once — inside the outcome. The works table records
    // the observation view, not a second conversation payload.
    assert_eq!(checkpoint.works.len(), 1);
    assert_eq!(checkpoint.works[0].work, work);
    assert_eq!(checkpoint.works[0].state, WorkState::Paused);
    assert!(checkpoint.works[0].finished.is_none());
    assert_eq!(
        checkpoint.submit_keys.len(),
        1,
        "the submit key is replayable"
    );
}

#[tokio::test]
async fn a_running_work_refuses_export_with_busy() {
    let gateway = GatedGateway::new("held", false);
    let session = idle_session("c1-running", gateway.clone());
    let handle = session.handle();
    let receipt = handle.submit(session_req("s1", "go")).expect("accepted");
    gateway.wait_entered().await;

    match session.checkpoint() {
        Err(SessionError::Busy { active }) => assert_eq!(active, receipt.work),
        other => panic!("expected Busy while the work runs, got {other:?}"),
    }

    gateway.release();
    let done = handle
        .wait(&receipt.work, Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(done.observation.state, WorkState::Finished);
    // Quiescent again — the export works.
    assert!(session.checkpoint().is_ok());
}

#[test]
fn a_faulted_session_refuses_export() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let gateway = GatedGateway::new("held", true);
    let session = idle_session("c1-faulted", gateway.clone());
    let handle = session.handle();
    let receipt = rt.block_on(async {
        let receipt = handle.submit(session_req("s1", "go")).unwrap();
        gateway.wait_entered().await;
        receipt
    });
    drop(rt);

    assert_eq!(
        handle.observe(&receipt.work).unwrap().state,
        WorkState::Faulted
    );
    assert!(
        matches!(session.checkpoint(), Err(SessionError::Faulted { .. })),
        "a faulted session has no complete outcome to save"
    );
}

#[tokio::test]
async fn a_closed_session_exports_closed_and_the_restore_keeps_it() {
    let (session, _work, _revision, checkpoint) =
        paused_checkpoint("c4-closed", SessionConfig::default()).await;
    assert!(!checkpoint.closed, "an open owner exports an open envelope");

    // Shutdown keeps the paused material readable and stops acceptance; the
    // export still works and records the closed state.
    session.shutdown().await;
    let checkpoint = session
        .checkpoint()
        .expect("a closed session still exports");
    assert!(checkpoint.closed);

    let restored = restore(
        checkpoint,
        RecordingGateway::scripted(vec![]),
        SessionConfig::default(),
    );
    let handle = restored.handle();
    // The closed state is preserved, not upgraded: a new request is refused,
    // an accepted one still replays its original receipt.
    match handle.submit(session_req("after", "nope")) {
        Err(SessionError::Closed) => {}
        other => panic!("expected Closed for a new request, got {other:?}"),
    }
    let receipt = handle
        .submit(session_req("pause", "go"))
        .expect("the accepted key replays after the closed restore");
    assert_eq!(receipt.work.turn_id, TurnId::new("c4-closed-work-0"));
    assert_eq!(receipt.accepted_revision, 0);
}

// ---- C2: replay, allocation, deadlines ----------------------------------------

#[tokio::test]
async fn restore_replays_submit_receipts_and_keeps_conflicts() {
    let gateway = RecordingGateway::repeating_last(vec![Ok(endturn_output("done"))]);
    let session = idle_session("c2-submit", gateway.clone());
    let handle = session.handle();
    let receipt = handle.submit(session_req("k1", "first")).expect("accepted");
    let done = handle
        .wait(&receipt.work, Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(done.observation.state, WorkState::Finished);

    let checkpoint = session.checkpoint().expect("exports when idle");
    let restored = restore(checkpoint, gateway.clone(), SessionConfig::default());
    let handle = restored.handle();

    // Same key, same parts: the original receipt, no second execution.
    let calls = gateway.recorded().len();
    assert_eq!(
        handle.submit(session_req("k1", "first")).expect("replays"),
        receipt
    );
    assert_eq!(
        gateway.recorded().len(),
        calls,
        "the replay executes nothing"
    );

    // Same key, different parts: a conflict, not a new work.
    let other = SubmitRequest {
        request_key: "k1".into(),
        parts: vec![ContentPart::Text(TextPayload::new("different"))],
    };
    assert!(matches!(handle.submit(other), Err(SessionError::Conflict)));

    // The allocation progress continues: the next work takes the next id,
    // and the old result stays observable under its original ref.
    let next = handle
        .submit(session_req("k2", "second"))
        .expect("accepted");
    assert_eq!(next.work.turn_id, TurnId::new("c2-submit-work-1"));
    let done = handle
        .wait(&next.work, Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(done.observation.state, WorkState::Finished);
    let old = handle.observe(&receipt.work).unwrap();
    assert_eq!(old.state, WorkState::Finished);
    assert!(
        matches!(old.finished, Some(FinishedKind::Completed { .. })),
        "the restored result is still observable"
    );
}

#[tokio::test]
async fn restore_replays_a_resume_key_without_reexecuting() {
    let (session, work, revision, _pre_save) =
        paused_checkpoint("c2-resume", SessionConfig::default()).await;
    let handle = session.handle();
    let paused = handle.observe(&work).unwrap();
    let Some(PausePoint::AwaitingApproval { prepared, .. }) = paused.paused else {
        panic!("the work needs approval");
    };
    let request = approve(prepared.awaiting);

    // Resume before the save: the key is accepted, the work runs to
    // completion, and the session goes idle again — exportable.
    let resumed = handle
        .resume(&work, revision, "r1".into(), request.clone())
        .expect("the paused work resumes");
    let done = handle.wait(&work, Duration::from_secs(5)).await.unwrap();
    assert_eq!(done.observation.state, WorkState::Finished);

    let checkpoint = session.checkpoint().expect("exports when idle");
    let restored = restore(
        checkpoint,
        RecordingGateway::scripted(vec![Ok(endturn_output("unused"))]),
        SessionConfig::default(),
    );
    let handle = restored.handle();

    // The same key with the same request replays the original resume
    // receipt without resuming anything.
    let replay = handle
        .resume(&work, revision, "r1".into(), request.clone())
        .expect("the resume key replays");
    assert_eq!(replay, resumed);
    // The same key with a different request is a conflict.
    let conflicting = ResumeRequest {
        decision: request.decision.clone(),
        inject: vec![TextPayload::new("extra")],
    };
    assert!(matches!(
        handle.resume(&work, revision, "r1".into(), conflicting),
        Err(SessionError::Conflict)
    ));
    assert_eq!(
        handle.observe(&work).unwrap().state,
        WorkState::Finished,
        "neither replay nor conflict rewrote the finished work"
    );
}

#[tokio::test]
async fn restore_executes_nothing_and_the_pause_needs_an_explicit_resume() {
    let (session, work, revision, checkpoint) =
        paused_checkpoint("c3-resume", SessionConfig::default()).await;
    let handle = session.handle();
    let paused = handle.observe(&work).unwrap();
    let Some(PausePoint::AwaitingApproval { prepared, .. }) = paused.paused else {
        panic!("the work needs approval");
    };
    let request = approve(prepared.awaiting);
    drop(session);
    drop(handle);

    let gateway2 = RecordingGateway::scripted(vec![Ok(endturn_output("resumed"))]);
    let restored = restore(checkpoint, gateway2.clone(), SessionConfig::default());
    let handle = restored.handle();

    // Restore itself is silent: no model, no tool.
    assert!(gateway2.recorded().is_empty());
    // The restored work is observable as Paused at the saved revision.
    let obs = handle.observe(&work).unwrap();
    assert_eq!(obs.state, WorkState::Paused);
    assert_eq!(obs.revision, revision);
    assert!(obs.paused.is_some(), "the pause point survives the restore");

    // The explicit resume continues the original continuation and completes
    // with exactly one further model call.
    handle
        .resume(&work, revision, "r1".into(), request)
        .expect("the restored pause resumes");
    let done = handle.wait(&work, Duration::from_secs(5)).await.unwrap();
    assert_eq!(done.end, WaitEnd::ReachedState);
    assert_eq!(done.observation.state, WorkState::Finished);
    assert_eq!(
        gateway2.recorded().len(),
        1,
        "the continuation, not a fresh turn, ran"
    );
}

#[tokio::test]
async fn restore_rederives_the_remaining_deadline_instead_of_regranting() {
    let config = SessionConfig {
        retained_work_capacity: 256,
        work_deadline: Some(Duration::from_millis(250)),
    };
    let (session, work, revision, checkpoint) =
        paused_checkpoint("c2-deadline", config.clone()).await;
    drop(session);

    // Sleep past the accept-time deadline while the work is paused.
    tokio::time::sleep(Duration::from_millis(600)).await;

    let gateway2 = RecordingGateway::scripted(vec![Ok(endturn_output("late"))]);
    let restored = restore(checkpoint, gateway2.clone(), config);
    let handle = restored.handle();

    // The resume is accepted (validation is not a deadline check), but the
    // restored remaining time is already spent: the driver stops the work
    // before dispatching anything.
    handle
        .resume(&work, revision, "r1".into(), approve(vec![awaiting_echo()]))
        .expect("a stale deadline does not reject the resume request itself");
    let done = handle.wait(&work, Duration::from_secs(5)).await.unwrap();
    assert_eq!(done.observation.state, WorkState::Finished);
    match done.observation.finished {
        Some(FinishedKind::Interrupted { cause, .. }) => assert_eq!(
            cause,
            TurnInterruption::TurnDeadlineExceeded,
            "the saved deadline still bounds the restored work"
        ),
        other => panic!("expected a deadline interruption, got {other:?}"),
    }
    assert!(
        gateway2.recorded().is_empty(),
        "the expired deadline dispatched no model call"
    );
}

// ---- C3: validation rejections -------------------------------------------------

#[tokio::test]
async fn restore_rejects_an_unsupported_version_and_returns_the_checkpoint() {
    let (_, _, _, checkpoint) = paused_checkpoint("c3-version", SessionConfig::default()).await;
    let mut value = serde_json::to_value(&checkpoint).unwrap();
    value["version"] = json!(99);
    let tampered: SessionCheckpoint = serde_json::from_value(value).unwrap();
    let as_saved = serde_json::to_value(&tampered).unwrap();

    let rejection = SessionCheckpoint::restore(
        tampered,
        Arc::new(runner_with(RecordingGateway::scripted(vec![]), Vec::new())),
        TurnRunOptions::default(),
        SessionConfig::default(),
    )
    .expect_err("an unsupported version is rejected");

    assert!(matches!(
        rejection.error,
        SessionError::InvalidCheckpoint(_)
    ));
    assert_eq!(
        serde_json::to_value(&rejection.checkpoint).unwrap(),
        as_saved,
        "the original materials come back untouched"
    );
    assert_eq!(rejection.config, SessionConfig::default());
}

#[tokio::test]
async fn restore_rejects_a_config_mismatch() {
    let (_, _, _, checkpoint) = paused_checkpoint("c4-config", SessionConfig::default()).await;
    let as_saved = serde_json::to_value(&checkpoint).unwrap();
    let mut swapped = TurnRunOptions::default();
    swapped.invocation.model = ModelRef::new("other-model");

    let rejection = SessionCheckpoint::restore(
        checkpoint,
        Arc::new(runner_with(RecordingGateway::scripted(vec![]), Vec::new())),
        swapped,
        SessionConfig::default(),
    )
    .expect_err("a swapped model is a configuration mismatch");

    assert!(matches!(rejection.error, SessionError::ConfigMismatch));
    assert_eq!(
        serde_json::to_value(&rejection.checkpoint).unwrap(),
        as_saved
    );
}

#[tokio::test]
async fn restore_rejects_inconsistent_material_and_returns_the_checkpoint() {
    // (a) the paused work is not registered
    let (_, _, _, checkpoint) = paused_checkpoint("c4-material", SessionConfig::default()).await;
    let mut value = serde_json::to_value(&checkpoint).unwrap();
    value["works"] = json!([]);
    let as_saved = value.clone();
    let tampered: SessionCheckpoint = serde_json::from_value(value).unwrap();
    let rejection = SessionCheckpoint::restore(
        tampered,
        Arc::new(runner_with(RecordingGateway::scripted(vec![]), Vec::new())),
        TurnRunOptions::default(),
        SessionConfig::default(),
    )
    .expect_err("an unregistered paused work is inconsistent");
    assert!(matches!(
        rejection.error,
        SessionError::InvalidCheckpoint(_)
    ));
    assert_eq!(
        serde_json::to_value(&rejection.checkpoint).unwrap(),
        as_saved
    );

    // (b) a submit receipt references an unknown work
    let gateway = RecordingGateway::repeating_last(vec![Ok(endturn_output("done"))]);
    let session = idle_session("c4-material-b", gateway.clone());
    let handle = session.handle();
    let receipt = handle.submit(session_req("k1", "first")).unwrap();
    let _ = handle
        .wait(&receipt.work, Duration::from_secs(5))
        .await
        .unwrap();
    let checkpoint = session.checkpoint().unwrap();
    let mut value = serde_json::to_value(&checkpoint).unwrap();
    value["works"] = json!([]);
    let tampered: SessionCheckpoint = serde_json::from_value(value).unwrap();
    let rejection = SessionCheckpoint::restore(
        tampered,
        Arc::new(runner_with(RecordingGateway::scripted(vec![]), Vec::new())),
        TurnRunOptions::default(),
        SessionConfig::default(),
    )
    .expect_err("a dangling receipt is inconsistent");
    assert!(matches!(
        rejection.error,
        SessionError::InvalidCheckpoint(_)
    ));

    // (c) the paused outcome's active turn does not match its registered work
    let (_, _, _, checkpoint) = paused_checkpoint("c4-material-c", SessionConfig::default()).await;
    let mut value = serde_json::to_value(&checkpoint).unwrap();
    value["phase"]["work"]["turn_id"] = json!("c4-material-c-work-99");
    let tampered: SessionCheckpoint = serde_json::from_value(value).unwrap();
    let rejection = SessionCheckpoint::restore(
        tampered,
        Arc::new(runner_with(RecordingGateway::scripted(vec![]), Vec::new())),
        TurnRunOptions::default(),
        SessionConfig::default(),
    )
    .expect_err("a mismatched active turn is inconsistent");
    assert!(matches!(
        rejection.error,
        SessionError::InvalidCheckpoint(_)
    ));
}

// ---- C4: from-history distinction, persistence ---------------------------------

#[tokio::test]
async fn opening_from_history_rebuilds_no_dedup_table() {
    let gateway = RecordingGateway::repeating_last(vec![Ok(endturn_output("done"))]);
    let session = idle_session("c4-history", gateway.clone());
    let handle = session.handle();
    let receipt = handle.submit(session_req("k1", "first")).expect("accepted");
    let _ = handle
        .wait(&receipt.work, Duration::from_secs(5))
        .await
        .unwrap();
    let checkpoint = session.checkpoint().unwrap();

    // The restore path: the same key replays the original receipt and runs
    // nothing.
    let restored = restore(
        checkpoint.clone(),
        gateway.clone(),
        SessionConfig::default(),
    );
    let calls = gateway.recorded().len();
    assert_eq!(
        restored
            .handle()
            .submit(session_req("k1", "first"))
            .unwrap(),
        receipt
    );
    assert_eq!(gateway.recorded().len(), calls, "the restore replays");

    // The from-history path: the same history as a fresh state knows no
    // request keys and no allocation progress — the key is accepted as a
    // new work and executes again.
    let CheckpointPhase::Idle { state } = checkpoint.phase else {
        panic!("the completed session exports idle");
    };
    let reopened = Session::new(
        state,
        Arc::new(runner_with(gateway.clone(), Vec::new())),
        TurnRunOptions::default(),
        SessionConfig::default(),
    )
    .expect("a completed history is an idle session base");
    let again = reopened
        .handle()
        .submit(session_req("k1", "first"))
        .expect("no dedup table exists");
    assert_ne!(
        again, receipt,
        "the reopened session accepted the key as a new work"
    );
    assert_ne!(
        again.work.turn_id, receipt.work.turn_id,
        "the reopened session does not restore the allocation progress either"
    );
    let done = reopened
        .handle()
        .wait(&again.work, Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(done.observation.state, WorkState::Finished);
    assert!(
        gateway.recorded().len() > calls,
        "the reopened work executed"
    );
}

#[tokio::test]
async fn the_envelope_survives_a_persistence_roundtrip() {
    let (session, work, revision, checkpoint) =
        paused_checkpoint("c2-roundtrip", SessionConfig::default()).await;
    drop(session);

    let bytes = serde_json::to_vec(&checkpoint).expect("the envelope serializes");
    let reloaded: SessionCheckpoint =
        serde_json::from_slice(&bytes).expect("the envelope deserializes");

    let gateway2 = RecordingGateway::scripted(vec![Ok(endturn_output("resumed"))]);
    let restored = restore(reloaded, gateway2.clone(), SessionConfig::default());
    let handle = restored.handle();

    // The reloaded envelope restores to the same observable pause...
    let obs = handle.observe(&work).unwrap();
    assert_eq!(obs.state, WorkState::Paused);
    assert_eq!(obs.revision, revision);
    // ...replays the saved submit key...
    assert_eq!(
        handle.submit(session_req("pause", "go")).unwrap().work,
        work
    );
    // ...and the explicit resume completes the original continuation.
    handle
        .resume(&work, revision, "r1".into(), approve(vec![awaiting_echo()]))
        .expect("the restored pause resumes");
    let done = handle.wait(&work, Duration::from_secs(5)).await.unwrap();
    assert_eq!(done.observation.state, WorkState::Finished);
    assert_eq!(gateway2.recorded().len(), 1);
}
