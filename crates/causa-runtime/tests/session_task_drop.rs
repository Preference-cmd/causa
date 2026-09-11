//! A runtime may discard a task before its first poll or while it is
//! suspended. Surviving session owners must see a fault and still shut down.

mod common;

use std::time::Duration;

use causa_runtime::{Session, SessionError, SessionHandle, WorkRef, WorkState};
use common::{
    GatedGateway, RecordingGateway, approve, endturn_output, idle_session, paused_work, session_req,
};

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

fn assert_faulted_and_shutdown(session: Session, handle: SessionHandle, work: WorkRef) {
    let fault = handle.observe(&work).unwrap();
    assert_eq!(fault.state, WorkState::Faulted);
    assert!(fault.fault.as_deref().unwrap().contains("task dropped"));
    assert!(fault.finished.is_none());
    assert!(fault.paused.is_none());
    assert!(matches!(
        handle.submit(session_req("later", "must not run")),
        Err(SessionError::Faulted { .. })
    ));
    runtime().block_on(async {
        let waited = handle.wait(&work, Duration::from_secs(1)).await.unwrap();
        assert_eq!(waited.observation.state, WorkState::Faulted);
        tokio::time::timeout(Duration::from_secs(1), session.shutdown())
            .await
            .expect("shutdown cannot wait for a worker that no longer exists");
    });
    assert_eq!(handle.observe(&work).unwrap().revision, fault.revision);
}

#[test]
fn runtime_drop_before_first_poll_faults_accepted_work() {
    let rt = runtime();
    let gateway = RecordingGateway::scripted(vec![Ok(endturn_output("unused"))]);
    let session = idle_session("unpolled", gateway.clone());
    let handle = session.handle();
    let receipt = {
        let _entered = rt.enter();
        handle.submit(session_req("first", "go")).unwrap()
    };
    assert_eq!(
        handle.observe(&receipt.work).unwrap().state,
        WorkState::Accepted
    );
    drop(rt);
    assert!(gateway.recorded().is_empty());
    assert_eq!(handle.submit(session_req("first", "go")).unwrap(), receipt);
    assert_faulted_and_shutdown(session, handle, receipt.work);
}

#[test]
fn runtime_drop_during_model_call_faults_running_work() {
    let rt = runtime();
    let gateway = GatedGateway::new("held", true);
    let session = idle_session("dropped-running", gateway.clone());
    let handle = session.handle();
    let receipt = rt.block_on(async {
        let receipt = handle.submit(session_req("first", "go")).unwrap();
        gateway.wait_entered().await;
        receipt
    });
    assert_eq!(
        handle.observe(&receipt.work).unwrap().state,
        WorkState::Running
    );
    drop(rt);
    assert_faulted_and_shutdown(session, handle, receipt.work);
}

#[test]
fn runtime_drop_before_resume_poll_faults_original_work() {
    let rt = runtime();
    let gateway = RecordingGateway::scripted(vec![Ok(common::tooluse_output(
        "approve?",
        "echo",
        serde_json::json!({"path":"changed.rs"}),
    ))]);
    let (session, handle, work) = rt.block_on(paused_work("dropped-resume", gateway));
    let paused = handle.observe(&work).unwrap();
    let Some(causa_runtime::PausePoint::AwaitingApproval { prepared, .. }) = paused.paused else {
        panic!("the work needs approval");
    };
    let request = approve(prepared.awaiting);
    let receipt = {
        let _entered = rt.enter();
        handle
            .resume(&work, paused.revision, "resume".into(), request.clone())
            .unwrap()
    };
    drop(rt);
    assert_eq!(
        handle
            .resume(&work, paused.revision, "resume".into(), request)
            .unwrap(),
        receipt
    );
    assert_faulted_and_shutdown(session, handle, work);
}
