//! Slice 8 Phase D acceptance — the offline harness coordinator:
//! multi-session create / lookup / `WorkRef` routing (D1), cross-session create
//! dedup and conflict (D2), trusted depth and finite run capacity (D3), and
//! independent child control, fact isolation, and collective shutdown (D4).
//!
//! The reference coordinator lives in `tests/common/coordinator.rs` and is
//! never part of the published runtime; this target only proves its
//! coordination semantics through direct calls.

mod common;

use std::sync::Arc;
use std::sync::Barrier;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use causa_kernel::{ContentPart, ConversationId, TextPayload, TurnId};
use causa_runtime::{
    CancelOutcome, ConversationState, FinishedKind, SessionCheckpoint, SessionConfig, SessionError,
    TurnInterruption, TurnRunOptions, WaitEnd, WorkRef, WorkState,
};
use common::coordinator::{
    Coordinator, CoordinatorConfig, CoordinatorError, CreateOrigin, CreateRequest, Profile,
    Profiles, SessionParts, profile_factory,
};
use common::{
    EchoTool, GatedGateway, RecordingGateway, SlowGateway, approve, awaiting_echo, endturn_output,
    runner_with, session_req, tooluse_output,
};

/// One text part, the shape create requests carry.
fn parts(text: &str) -> Vec<ContentPart> {
    vec![ContentPart::Text(TextPayload::new(text))]
}

fn cfg(max_parallel: usize, max_depth: u32) -> CoordinatorConfig {
    CoordinatorConfig {
        max_parallel,
        max_depth,
    }
}

fn harness(config: CoordinatorConfig, profiles: Profiles) -> Arc<Coordinator> {
    Arc::new(Coordinator::new(config, profile_factory(profiles)))
}

/// The envelope as comparable JSON. No deadlines are configured in these
/// tests, so an export is stable (Phase C records that a re-derived deadline
/// re-anchors and is therefore not byte-equal).
fn json(checkpoint: &SessionCheckpoint) -> serde_json::Value {
    serde_json::to_value(checkpoint).expect("the envelope serializes")
}

/// A later work enters the capacity accounting when the coordinator resumes it.
#[tokio::test]
async fn d3_resuming_a_later_work_counts_toward_capacity() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output(
            "first work paused",
            "echo",
            serde_json::json!({}),
        )),
        Ok(endturn_output("first work finished")),
        Ok(tooluse_output(
            "second work paused",
            "echo",
            serde_json::json!({}),
        )),
        Ok(endturn_output("second work finished")),
    ]);
    let mut profiles = Profiles::new();
    profiles.insert("a", Profile::pausing(gateway, vec![Arc::new(EchoTool)]));
    profiles.insert(
        "b",
        Profile::completing(RecordingGateway::scripted(vec![Ok(endturn_output("b"))])),
    );
    let coordinator = harness(cfg(1, 2), profiles);
    let first = coordinator
        .create(CreateRequest::root("a", "first", parts("first"), "a"))
        .unwrap();
    let first_paused = coordinator
        .wait(&first.work, Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(first_paused.observation.state, WorkState::Paused);
    let first_resume = coordinator
        .resume(
            &first.work,
            first_paused.observation.revision,
            "resume-first",
            approve(vec![awaiting_echo()]),
        )
        .unwrap();
    let finished = coordinator
        .wait(&first.work, Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(finished.observation.state, WorkState::Finished);
    let next = coordinator
        .route(&first.work)
        .unwrap()
        .submit(session_req("next", "next"))
        .unwrap();
    let paused = coordinator
        .wait(&next.work, Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(paused.observation.state, WorkState::Paused);
    coordinator
        .resume(
            &next.work,
            paused.observation.revision,
            "resume-next",
            approve(vec![awaiting_echo()]),
        )
        .unwrap();
    // No await on this current-thread runtime: admission has marked the work
    // Running, and its worker cannot finish before the capacity assertion.
    assert_eq!(
        coordinator.observe(&next.work).unwrap().state,
        WorkState::Running
    );
    assert_eq!(
        coordinator
            .resume(
                &first.work,
                first_paused.observation.revision,
                "resume-first",
                approve(vec![awaiting_echo()]),
            )
            .unwrap(),
        first_resume,
        "replaying the earlier work must not replace the currently counted work",
    );
    let other = CreateRequest::root("b", "other", parts("other"), "b");
    assert_eq!(
        coordinator.create(other.clone()),
        Err(CoordinatorError::CapacityExceeded)
    );
    let finished = coordinator
        .wait(&next.work, Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(finished.observation.state, WorkState::Finished);
    coordinator
        .create(other)
        .expect("completion frees capacity without consuming the refused key");
    assert_eq!(
        coordinator.depth_of(&first.work),
        Some(0),
        "capacity tracking preserves the original creation relation"
    );
    coordinator.shutdown().await;
}

/// Replaying an accepted resume consumes no capacity, even after another pause.
#[tokio::test]
async fn d3_resume_replays_before_checking_capacity() {
    let a = RecordingGateway::scripted(vec![
        Ok(tooluse_output("first pause", "echo", serde_json::json!({}))),
        Ok(tooluse_output(
            "second pause",
            "echo",
            serde_json::json!({}),
        )),
    ]);
    let gate = GatedGateway::new("b", true);
    let mut profiles = Profiles::new();
    profiles.insert("a", Profile::pausing(a.clone(), vec![Arc::new(EchoTool)]));
    profiles.insert("b", Profile::completing(gate.clone()));
    let coordinator = harness(cfg(1, 2), profiles);
    let work = coordinator
        .create(CreateRequest::root("a", "a", parts("a"), "a"))
        .unwrap()
        .work;
    let paused = coordinator
        .wait(&work, Duration::from_secs(5))
        .await
        .unwrap();
    let revision = paused.observation.revision;
    let request = approve(vec![awaiting_echo()]);
    let original = coordinator
        .resume(&work, revision, "resume", request.clone())
        .unwrap();
    let paused_again = coordinator
        .wait(&work, Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(paused_again.observation.state, WorkState::Paused);
    coordinator
        .create(CreateRequest::root("b", "b", parts("b"), "b"))
        .unwrap();
    gate.wait_entered().await;
    assert_eq!(
        coordinator
            .resume(&work, revision, "resume", request.clone())
            .unwrap(),
        original
    );
    assert_eq!(
        coordinator.resume(&work, revision + 1, "resume", request.clone()),
        Err(CoordinatorError::Session(SessionError::Conflict))
    );
    assert_eq!(
        coordinator.resume(
            &work,
            paused_again.observation.revision,
            "new-resume",
            request
        ),
        Err(CoordinatorError::CapacityExceeded)
    );
    assert_eq!(
        coordinator.observe(&work).unwrap().revision,
        paused_again.observation.revision
    );
    assert_eq!(
        a.recorded().len(),
        2,
        "replay and refusals execute no further model calls"
    );
    coordinator.shutdown().await;
}

/// The minimal consumer gate: the create / route / observe / wait / shutdown
/// signatures compile and the closed loop runs before the acceptance suite.
#[tokio::test]
async fn gate_minimal_consumer_compiles_and_runs() {
    let gateway = RecordingGateway::scripted(vec![Ok(endturn_output("done"))]);
    let mut profiles = Profiles::new();
    profiles.insert("scripted", Profile::completing(gateway));

    let coordinator = harness(cfg(2, 2), profiles);

    let receipt = coordinator
        .create(CreateRequest::root("c1", "k1", parts("hi"), "scripted"))
        .expect("an idle profile is admitted");
    let _handle = coordinator.route(&receipt.work).expect("routed");
    let observation = coordinator.observe(&receipt.work).expect("observable");
    assert!(matches!(
        observation.state,
        WorkState::Accepted | WorkState::Running | WorkState::Finished
    ));
    let waited = coordinator
        .wait(&receipt.work, Duration::from_secs(5))
        .await
        .expect("observable");
    assert_eq!(waited.end, WaitEnd::ReachedState);
    coordinator.shutdown().await;
}

// ---- D1: the harness holds several sessions and routes by WorkRef ----------

#[tokio::test]
async fn d1_create_admits_the_initial_work_and_routes_by_work_ref() {
    let gateway = RecordingGateway::scripted(vec![Ok(endturn_output("done"))]);
    let mut profiles = Profiles::new();
    profiles.insert("done", Profile::completing(gateway.clone()));
    let coordinator = harness(cfg(2, 2), profiles);

    let receipt = coordinator
        .create(CreateRequest::root("c1", "k1", parts("hello"), "done"))
        .expect("the idle profile is admitted");
    assert_eq!(receipt.work.conversation_id.0, "c1");
    assert_eq!(
        receipt.accepted_revision, 0,
        "the returned receipt is the initial submit's acceptance"
    );

    let handle = coordinator.route(&receipt.work).expect("routed by WorkRef");
    assert_eq!(handle.id().0, "c1");
    assert_eq!(
        coordinator.observe(&receipt.work).expect("observable").work,
        receipt.work
    );

    let waited = coordinator
        .wait(&receipt.work, Duration::from_secs(5))
        .await
        .expect("observable");
    assert_eq!(waited.end, WaitEnd::ReachedState);
    assert!(
        matches!(
            waited.observation.finished,
            Some(FinishedKind::Completed { .. })
        ),
        "the initial task completed: {:?}",
        waited.observation.finished
    );
    assert_eq!(
        gateway.recorded().len(),
        1,
        "the admitted work really ran the initial task"
    );
    coordinator.shutdown().await;
}

#[tokio::test]
async fn d1_the_receipt_outlives_the_create_scope() {
    let gateway = RecordingGateway::scripted(vec![Ok(endturn_output("done"))]);
    let mut profiles = Profiles::new();
    profiles.insert("done", Profile::completing(gateway));
    let coordinator = harness(cfg(2, 2), profiles);

    // The receipt is dropped at the end of this scope; the work reference and
    // the session must survive it (no create/wait future owns the session).
    let work = {
        let receipt = coordinator
            .create(CreateRequest::root("c1", "k1", parts("hello"), "done"))
            .expect("admitted");
        receipt.work
    };

    let by_id = coordinator
        .handle(&ConversationId("c1".into()))
        .expect("the session stays registered");
    assert_eq!(by_id.id().0, "c1");
    let waited = coordinator
        .wait(&work, Duration::from_secs(5))
        .await
        .expect("observable");
    assert_eq!(waited.observation.state, WorkState::Finished);
    coordinator.shutdown().await;
}

#[tokio::test]
async fn d1_routing_is_per_conversation() {
    let a = RecordingGateway::scripted(vec![Ok(endturn_output("a"))]);
    let b = RecordingGateway::scripted(vec![Ok(endturn_output("b"))]);
    let mut profiles = Profiles::new();
    profiles.insert("a", Profile::completing(a));
    profiles.insert("b", Profile::completing(b));
    let coordinator = harness(cfg(4, 2), profiles);

    let a_receipt = coordinator
        .create(CreateRequest::root("ca", "ka", parts("a"), "a"))
        .expect("admitted");
    let b_receipt = coordinator
        .create(CreateRequest::root("cb", "kb", parts("b"), "b"))
        .expect("admitted");
    assert_eq!(a_receipt.work.conversation_id.0, "ca");
    assert_eq!(b_receipt.work.conversation_id.0, "cb");

    // An unregistered conversation does not route anywhere.
    let unknown = WorkRef {
        conversation_id: ConversationId("nope".into()),
        turn_id: a_receipt.work.turn_id.clone(),
    };
    let error = coordinator
        .route(&unknown)
        .expect_err("unregistered conversations do not route");
    assert_eq!(
        error,
        CoordinatorError::UnknownConversation(ConversationId("nope".into()))
    );

    // A known conversation with an unknown turn routes to that session, which
    // owns the work table and rejects the ref itself — the coordinator keeps
    // no second copy.
    let foreign_turn = WorkRef {
        conversation_id: ConversationId("ca".into()),
        turn_id: TurnId::new("bogus"),
    };
    let handle = coordinator
        .route(&foreign_turn)
        .expect("routing is by conversation");
    assert_eq!(handle.id().0, "ca");
    let error = coordinator
        .observe(&foreign_turn)
        .expect_err("the session owns the work table");
    assert_eq!(
        error,
        CoordinatorError::Session(SessionError::NotFound(foreign_turn.clone()))
    );

    let _ = coordinator
        .wait(&a_receipt.work, Duration::from_secs(5))
        .await
        .expect("observable");
    let _ = coordinator
        .wait(&b_receipt.work, Duration::from_secs(5))
        .await
        .expect("observable");
    coordinator.shutdown().await;
}

// ---- D2: cross-session create dedup and conflict ---------------------------

#[tokio::test]
async fn d2_repeated_create_replays_the_original_receipt() {
    let gateway = RecordingGateway::scripted(vec![Ok(endturn_output("done"))]);
    let mut profiles = Profiles::new();
    profiles.insert("done", Profile::completing(gateway.clone()));
    let coordinator = harness(cfg(2, 2), profiles);

    let request = CreateRequest::root("c1", "k1", parts("hello"), "done");
    let first = coordinator.create(request.clone()).expect("admitted");
    let replay = coordinator
        .create(request)
        .expect("a lost-receipt retry replays the original receipt");

    assert_eq!(replay, first);
    assert_eq!(
        coordinator.session_count(),
        1,
        "the retry creates no second session"
    );
    let _ = coordinator
        .wait(&first.work, Duration::from_secs(5))
        .await
        .expect("observable");
    assert_eq!(gateway.recorded().len(), 1, "the initial task runs once");
    coordinator.shutdown().await;
}

#[tokio::test]
async fn d2_same_key_different_args_conflicts() {
    let done = RecordingGateway::scripted(vec![Ok(endturn_output("done"))]);
    let other = RecordingGateway::scripted(vec![Ok(endturn_output("other"))]);
    let mut profiles = Profiles::new();
    profiles.insert("done", Profile::completing(done));
    profiles.insert("other", Profile::completing(other));
    let coordinator = harness(cfg(4, 2), profiles);

    let base = CreateRequest::root("c1", "k1", parts("hello"), "done");
    let first = coordinator.create(base.clone()).expect("admitted");

    // A different conversation, different parts, a different selection, and a
    // different origin are all different arguments under the same key.
    let variants = vec![
        CreateRequest::root("c9", "k1", parts("hello"), "done"),
        CreateRequest::root("c1", "k1", parts("changed"), "done"),
        CreateRequest::root("c1", "k1", parts("hello"), "other"),
        CreateRequest {
            request_key: "k1".into(),
            conversation_id: ConversationId("c1".into()),
            parts: parts("hello"),
            origin: CreateOrigin::Child {
                parent: first.work.clone(),
            },
            selection: "done".into(),
        },
    ];
    for variant in variants {
        assert_eq!(
            coordinator.create(variant),
            Err(CoordinatorError::Conflict),
            "the same key with different arguments must conflict"
        );
    }
    assert_eq!(coordinator.session_count(), 1, "a conflict creates nothing");
    assert_eq!(
        coordinator
            .create(base)
            .expect("the original still replays"),
        first
    );
    coordinator.shutdown().await;
}

#[tokio::test]
async fn d2_concurrent_same_key_creates_once() {
    let gateway = RecordingGateway::scripted(vec![Ok(endturn_output("done"))]);
    let mut profiles = Profiles::new();
    profiles.insert("done", Profile::completing(gateway.clone()));
    let coordinator = harness(cfg(4, 2), profiles);

    let request = CreateRequest::root("c1", "k1", parts("hello"), "done");
    let barrier = Arc::new(Barrier::new(2));
    let mut tasks = Vec::new();
    for _ in 0..2 {
        let coordinator = Arc::clone(&coordinator);
        let request = request.clone();
        let barrier = Arc::clone(&barrier);
        tasks.push(tokio::task::spawn_blocking(move || {
            barrier.wait();
            coordinator.create(request)
        }));
    }
    let second = tasks.pop().expect("two tasks");
    let first = tasks.pop().expect("two tasks");
    let (first, second) = tokio::join!(first, second);
    let first = first
        .expect("the blocking task joins")
        .expect("one create wins");
    let second = second
        .expect("the blocking task joins")
        .expect("the other replays the receipt");

    assert_eq!(first, second, "both callers observe the same acceptance");
    assert_eq!(coordinator.session_count(), 1);
    let _ = coordinator
        .wait(&first.work, Duration::from_secs(5))
        .await
        .expect("observable");
    assert_eq!(gateway.recorded().len(), 1, "the initial task runs once");
    coordinator.shutdown().await;
}

#[tokio::test]
async fn d2_rejected_create_does_not_consume_the_key() {
    let gate = GatedGateway::new("late", false);
    let done = RecordingGateway::scripted(vec![Ok(endturn_output("done"))]);
    let mut profiles = Profiles::new();
    profiles.insert("gated", Profile::completing(gate.clone()));
    profiles.insert("done", Profile::completing(done));
    let coordinator = harness(cfg(1, 1), profiles);

    // A capacity refusal frees the key: the same request is admitted once the
    // slot frees.
    let root = coordinator
        .create(CreateRequest::root("ca", "ka", parts("a"), "gated"))
        .expect("admitted");
    gate.wait_entered().await;
    assert_eq!(
        coordinator.create(CreateRequest::root("cb", "kb", parts("b"), "done")),
        Err(CoordinatorError::CapacityExceeded)
    );
    gate.release();
    let _ = coordinator
        .wait(&root.work, Duration::from_secs(5))
        .await
        .expect("observable");
    let admitted = coordinator
        .create(CreateRequest::root("cb", "kb", parts("b"), "done"))
        .expect("the capacity refusal left the key free");
    let _ = coordinator
        .wait(&admitted.work, Duration::from_secs(5))
        .await
        .expect("observable");

    // A depth refusal frees it too, even when the retry changes the origin.
    let child = coordinator
        .create(CreateRequest::child(
            "cc",
            "kc",
            parts("c"),
            "done",
            &admitted.work,
        ))
        .expect("depth 1 is allowed");
    let _ = coordinator
        .wait(&child.work, Duration::from_secs(5))
        .await
        .expect("observable");
    assert_eq!(
        coordinator.create(CreateRequest::child(
            "cd",
            "kd",
            parts("d"),
            "done",
            &child.work
        )),
        Err(CoordinatorError::DepthExceeded { depth: 2, max: 1 })
    );
    let retried = coordinator
        .create(CreateRequest::root("cd", "kd", parts("d"), "done"))
        .expect("the depth refusal left the key free");
    assert_eq!(retried.work.conversation_id.0, "cd");
    coordinator.shutdown().await;
}

// ---- D3: finite capacity, trusted depth, and lock discipline ---------------

#[tokio::test]
async fn d3_capacity_is_checked_before_acceptance() {
    let builds = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&builds);
    let gate = GatedGateway::new("late", false);
    let inner = profile_factory({
        let mut profiles = Profiles::new();
        profiles.insert("gated", Profile::completing(gate.clone()));
        profiles
    });
    let coordinator = Arc::new(Coordinator::new(
        cfg(1, 2),
        move |request: &CreateRequest| {
            counted.fetch_add(1, Ordering::SeqCst);
            inner(request)
        },
    ));

    let root = coordinator
        .create(CreateRequest::root("ca", "ka", parts("a"), "gated"))
        .expect("admitted");
    // The gated gateway parks inside the model call, so the work is Running.
    gate.wait_entered().await;
    assert_eq!(builds.load(Ordering::SeqCst), 1);
    assert_eq!(
        coordinator.observe(&root.work).expect("observable").state,
        WorkState::Running
    );

    // The refusal happens before the harness assembles anything.
    let refused = coordinator.create(CreateRequest::root("cb", "kb", parts("b"), "gated"));
    assert_eq!(refused, Err(CoordinatorError::CapacityExceeded));
    assert_eq!(
        builds.load(Ordering::SeqCst),
        1,
        "a capacity refusal never consults the factory"
    );
    assert_eq!(coordinator.session_count(), 1);
    let error = coordinator
        .handle(&ConversationId("cb".into()))
        .expect_err("no session was registered");
    assert_eq!(
        error,
        CoordinatorError::UnknownConversation(ConversationId("cb".into()))
    );

    // The gated work ignores the stop token by construction; release it before
    // the collective shutdown so the session can settle.
    gate.release();
    let _ = coordinator
        .wait(&root.work, Duration::from_secs(5))
        .await
        .expect("observable");
    coordinator.shutdown().await;
}

#[tokio::test]
async fn d3_paused_frees_capacity_and_resume_reapplies() {
    // One slot, two pausable profiles, one gated profile.
    let a = RecordingGateway::scripted(vec![
        Ok(tooluse_output("a", "echo", serde_json::json!({}))),
        Ok(endturn_output("a done")),
    ]);
    let b =
        RecordingGateway::scripted(vec![Ok(tooluse_output("b", "echo", serde_json::json!({})))]);
    let gate = GatedGateway::new("c", false);
    let mut profiles = Profiles::new();
    profiles.insert("paused-a", Profile::pausing(a, vec![Arc::new(EchoTool)]));
    profiles.insert("paused-b", Profile::pausing(b, vec![Arc::new(EchoTool)]));
    profiles.insert("gated", Profile::completing(gate.clone()));
    let coordinator = harness(cfg(1, 2), profiles);

    // A pauses and frees the only slot.
    let a_receipt = coordinator
        .create(CreateRequest::root("ca", "ka", parts("a"), "paused-a"))
        .expect("admitted");
    let a_paused = coordinator
        .wait(&a_receipt.work, Duration::from_secs(5))
        .await
        .expect("observable");
    assert_eq!(a_paused.observation.state, WorkState::Paused);

    // B is admitted while A is paused...
    let b_receipt = coordinator
        .create(CreateRequest::root("cb", "kb", parts("b"), "paused-b"))
        .expect("a paused work frees its slot");
    let b_paused = coordinator
        .wait(&b_receipt.work, Duration::from_secs(5))
        .await
        .expect("observable");
    assert_eq!(b_paused.observation.state, WorkState::Paused);

    // ...and C takes the slot while both are paused.
    let c_receipt = coordinator
        .create(CreateRequest::root("cc", "kc", parts("c"), "gated"))
        .expect("paused works hold no run capacity");
    gate.wait_entered().await;

    // Resuming re-applies for capacity: refused while C runs, and the refusal
    // leaves A paused without consuming the key.
    let refused = coordinator.resume(
        &a_receipt.work,
        a_paused.observation.revision,
        "ra",
        approve(vec![awaiting_echo()]),
    );
    assert_eq!(refused, Err(CoordinatorError::CapacityExceeded));
    assert_eq!(
        coordinator
            .observe(&a_receipt.work)
            .expect("observable")
            .state,
        WorkState::Paused,
        "a capacity refusal keeps the work paused"
    );

    // Releasing C frees the slot; the same key now resumes A.
    gate.release();
    let _ = coordinator
        .wait(&c_receipt.work, Duration::from_secs(5))
        .await
        .expect("observable");
    let resumed = coordinator
        .resume(
            &a_receipt.work,
            a_paused.observation.revision,
            "ra",
            approve(vec![awaiting_echo()]),
        )
        .expect("the capacity refusal did not consume the resume key");
    assert_eq!(resumed.work, a_receipt.work);
    let a_finished = coordinator
        .wait(&a_receipt.work, Duration::from_secs(5))
        .await
        .expect("observable");
    assert!(matches!(
        a_finished.observation.finished,
        Some(FinishedKind::Completed { .. })
    ));

    // B was never touched by any of this.
    assert_eq!(
        coordinator
            .observe(&b_receipt.work)
            .expect("observable")
            .state,
        WorkState::Paused
    );
    coordinator.shutdown().await;
}

#[tokio::test]
async fn d3_depth_comes_from_trusted_relations() {
    let gateway = RecordingGateway::repeating_last(vec![Ok(endturn_output("done"))]);
    let mut profiles = Profiles::new();
    profiles.insert("done", Profile::completing(gateway));
    let coordinator = harness(cfg(8, 2), profiles);

    let root = coordinator
        .create(CreateRequest::root("c0", "k0", parts("0"), "done"))
        .expect("admitted");
    assert_eq!(coordinator.depth_of(&root.work), Some(0));
    let child = coordinator
        .create(CreateRequest::child(
            "c1",
            "k1",
            parts("1"),
            "done",
            &root.work,
        ))
        .expect("depth 1 is allowed");
    assert_eq!(coordinator.depth_of(&child.work), Some(1));
    let grandchild = coordinator
        .create(CreateRequest::child(
            "c2",
            "k2",
            parts("2"),
            "done",
            &child.work,
        ))
        .expect("depth 2 is allowed");
    assert_eq!(coordinator.depth_of(&grandchild.work), Some(2));

    assert_eq!(
        coordinator.create(CreateRequest::child(
            "c3",
            "k3",
            parts("3"),
            "done",
            &grandchild.work
        )),
        Err(CoordinatorError::DepthExceeded { depth: 3, max: 2 })
    );
    assert_eq!(
        coordinator.session_count(),
        3,
        "a depth refusal creates nothing"
    );

    // The depth is derived from the coordinator's own relation, so a parent it
    // never admitted is refused rather than guessed.
    let ghost = WorkRef {
        conversation_id: ConversationId("ghost".into()),
        turn_id: TurnId::new("t"),
    };
    assert_eq!(
        coordinator.create(CreateRequest::child("c4", "k4", parts("4"), "done", &ghost)),
        Err(CoordinatorError::UnknownParent(ghost.clone()))
    );
    let wrong_turn = WorkRef {
        conversation_id: ConversationId("c0".into()),
        turn_id: TurnId::new("bogus"),
    };
    assert_eq!(
        coordinator.create(CreateRequest::child(
            "c4",
            "k4",
            parts("4"),
            "done",
            &wrong_turn
        )),
        Err(CoordinatorError::UnknownParent(wrong_turn.clone()))
    );

    // Both refusals left the key free, so a root create still succeeds.
    let admitted = coordinator
        .create(CreateRequest::root("c4", "k4", parts("4"), "done"))
        .expect("the refused keys are free");
    assert_eq!(coordinator.depth_of(&admitted.work), Some(0));
    coordinator.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn d3_waiters_do_not_hold_the_coordination_lock() {
    let gate = GatedGateway::new("late", false);
    let done = RecordingGateway::scripted(vec![Ok(endturn_output("done"))]);
    let mut profiles = Profiles::new();
    profiles.insert("gated", Profile::completing(gate.clone()));
    profiles.insert("done", Profile::completing(done));
    let coordinator = harness(cfg(2, 2), profiles);

    let a = coordinator
        .create(CreateRequest::root("ca", "ka", parts("a"), "gated"))
        .expect("admitted");
    gate.wait_entered().await;

    // Park a coordinator wait on A and know it entered the wait path.
    let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
    let waiter = {
        let coordinator = Arc::clone(&coordinator);
        let work = a.work.clone();
        tokio::spawn(async move {
            // Poll the wait at least once before announcing entry: in a
            // lock-holding implementation the guard is already held at that
            // point, so the probe cannot race an unpolled task.
            let waiting = coordinator.wait(&work, Duration::from_secs(5));
            let mut waiting = std::pin::pin!(waiting);
            std::future::poll_fn(|cx| {
                let _ = waiting.as_mut().poll(cx);
                std::task::Poll::Ready(())
            })
            .await;
            let _ = entered_tx.send(());
            waiting.await
        })
    };
    entered_rx.await.expect("the waiter started");
    tokio::task::yield_now().await;

    // Admission from another thread returns while the wait is still pending:
    // the coordination lock is not held across it.
    let admitted = {
        let coordinator = Arc::clone(&coordinator);
        let created = tokio::time::timeout(
            Duration::from_secs(1),
            tokio::task::spawn_blocking(move || {
                coordinator.create(CreateRequest::root("cb", "kb", parts("b"), "done"))
            }),
        )
        .await;
        match created {
            Ok(joined) => joined
                .expect("the blocking task joins")
                .expect("capacity admits the second session"),
            Err(_) => {
                // Let the parked work finish so the runtime can drain the
                // blocking task, then fail loudly.
                gate.release();
                let _ = waiter.await;
                panic!(
                    "create was blocked by a pending waiter: the coordination lock is held across it"
                );
            }
        }
    };
    assert_eq!(admitted.work.conversation_id.0, "cb");

    gate.release();
    let a_waited = waiter
        .await
        .expect("the waiter task joins")
        .expect("observable");
    assert_eq!(a_waited.observation.state, WorkState::Finished);
    coordinator.shutdown().await;
}

// ---- D4: independent child control, fact isolation, collective shutdown ----

#[tokio::test]
async fn d4_parent_completion_does_not_cancel_the_child() {
    let parent = RecordingGateway::scripted(vec![Ok(endturn_output("parent done"))]);
    let child = GatedGateway::new("child done", false);
    let mut profiles = Profiles::new();
    profiles.insert("parent", Profile::completing(parent));
    profiles.insert("child", Profile::completing(child.clone()));
    let coordinator = harness(cfg(2, 2), profiles);

    let parent_receipt = coordinator
        .create(CreateRequest::root("cp", "kp", parts("p"), "parent"))
        .expect("admitted");
    let _ = coordinator
        .wait(&parent_receipt.work, Duration::from_secs(5))
        .await
        .expect("observable");

    // The child is created after the parent finished and runs on its own.
    let child_receipt = coordinator
        .create(CreateRequest::child(
            "cc",
            "kc",
            parts("c"),
            "child",
            &parent_receipt.work,
        ))
        .expect("a finished parent still anchors a child");
    child.wait_entered().await;
    assert_eq!(
        coordinator
            .observe(&child_receipt.work)
            .expect("observable")
            .state,
        WorkState::Running
    );

    let parent_observation = coordinator
        .observe(&parent_receipt.work)
        .expect("observable");
    assert!(matches!(
        parent_observation.finished,
        Some(FinishedKind::Completed { .. })
    ));

    child.release();
    let child_finished = coordinator
        .wait(&child_receipt.work, Duration::from_secs(5))
        .await
        .expect("observable");
    assert!(matches!(
        child_finished.observation.finished,
        Some(FinishedKind::Completed { .. })
    ));

    // The parent can still take its own next work.
    let handle = coordinator.route(&parent_receipt.work).expect("routed");
    let next = handle
        .submit(session_req("kp2", "again"))
        .expect("the parent is idle again");
    assert_ne!(next.work.turn_id, parent_receipt.work.turn_id);
    coordinator.shutdown().await;
}

#[tokio::test]
async fn d4_parent_cancel_does_not_cancel_the_child() {
    let parent = GatedGateway::new("parent", true);
    let child = GatedGateway::new("child", true);
    let mut profiles = Profiles::new();
    profiles.insert("parent", Profile::completing(parent.clone()));
    profiles.insert("child", Profile::completing(child.clone()));
    let coordinator = harness(cfg(2, 2), profiles);

    let parent_receipt = coordinator
        .create(CreateRequest::root("cp", "kp", parts("p"), "parent"))
        .expect("admitted");
    parent.wait_entered().await;
    let child_receipt = coordinator
        .create(CreateRequest::child(
            "cc",
            "kc",
            parts("c"),
            "child",
            &parent_receipt.work,
        ))
        .expect("admitted");
    child.wait_entered().await;

    // Cancelling the parent settles only the parent's own control domain.
    let cancel = coordinator
        .route(&parent_receipt.work)
        .expect("routed")
        .cancel(&parent_receipt.work, "cancel-parent".to_string())
        .expect("the cancel is accepted");
    assert_eq!(cancel.outcome, CancelOutcome::Signalled);
    let parent_finished = coordinator
        .wait(&parent_receipt.work, Duration::from_secs(5))
        .await
        .expect("observable");
    assert!(matches!(
        parent_finished.observation.finished,
        Some(FinishedKind::Interrupted { .. })
    ));

    // The child is untouched and still completes.
    assert_eq!(
        coordinator
            .observe(&child_receipt.work)
            .expect("observable")
            .state,
        WorkState::Running
    );
    child.release();
    let child_finished = coordinator
        .wait(&child_receipt.work, Duration::from_secs(5))
        .await
        .expect("observable");
    assert!(matches!(
        child_finished.observation.finished,
        Some(FinishedKind::Completed { .. })
    ));
    coordinator.shutdown().await;
}

#[tokio::test]
async fn d4_dropping_a_waiter_does_not_cancel_the_child() {
    let child = GatedGateway::new("child done", false);
    let mut profiles = Profiles::new();
    profiles.insert("child", Profile::completing(child.clone()));
    let coordinator = harness(cfg(2, 2), profiles);

    let receipt = coordinator
        .create(CreateRequest::root("cc", "kc", parts("c"), "child"))
        .expect("admitted");
    child.wait_entered().await;

    // A cancelled (dropped) waiter changes nothing about the work.
    let dropped = tokio::time::timeout(
        Duration::from_millis(20),
        coordinator.wait(&receipt.work, Duration::from_secs(5)),
    )
    .await;
    assert!(dropped.is_err(), "the outer timeout drops the wait future");
    assert_eq!(
        coordinator
            .observe(&receipt.work)
            .expect("observable")
            .state,
        WorkState::Running
    );

    child.release();
    let finished = coordinator
        .wait(&receipt.work, Duration::from_secs(5))
        .await
        .expect("observable");
    assert!(matches!(
        finished.observation.finished,
        Some(FinishedKind::Completed { .. })
    ));
    coordinator.shutdown().await;
}

#[tokio::test]
async fn d4_child_activity_never_rewrites_the_parent() {
    let parent = RecordingGateway::scripted(vec![Ok(endturn_output("parent done"))]);
    let child = RecordingGateway::scripted(vec![
        Ok(tooluse_output("child", "echo", serde_json::json!({}))),
        Ok(endturn_output("child done")),
    ]);
    let mut profiles = Profiles::new();
    profiles.insert("parent", Profile::completing(parent));
    profiles.insert("child", Profile::pausing(child, vec![Arc::new(EchoTool)]));
    let coordinator = harness(cfg(2, 2), profiles);

    let parent_receipt = coordinator
        .create(CreateRequest::root("cp", "kp", parts("p"), "parent"))
        .expect("admitted");
    let _ = coordinator
        .wait(&parent_receipt.work, Duration::from_secs(5))
        .await
        .expect("observable");
    let parent_id = ConversationId("cp".into());
    let before = coordinator
        .checkpoint(&parent_id)
        .expect("the finished parent exports");

    // The child pauses; the parent's saved facts are untouched.
    let child_receipt = coordinator
        .create(CreateRequest::child(
            "cc",
            "kc",
            parts("c"),
            "child",
            &parent_receipt.work,
        ))
        .expect("admitted");
    let paused = coordinator
        .wait(&child_receipt.work, Duration::from_secs(5))
        .await
        .expect("observable");
    assert_eq!(paused.observation.state, WorkState::Paused);
    let after_pause = coordinator.checkpoint(&parent_id).expect("exports");
    assert_eq!(
        json(&before),
        json(&after_pause),
        "a child pause must not rewrite the parent"
    );

    // Resuming and completing the child changes nothing either.
    let resumed = coordinator
        .resume(
            &child_receipt.work,
            paused.observation.revision,
            "rc",
            approve(vec![awaiting_echo()]),
        )
        .expect("resumed");
    assert_eq!(resumed.work, child_receipt.work);
    let _ = coordinator
        .wait(&child_receipt.work, Duration::from_secs(5))
        .await
        .expect("observable");
    let after_child = coordinator.checkpoint(&parent_id).expect("exports");
    assert_eq!(
        json(&before),
        json(&after_child),
        "child completion must not rewrite the parent"
    );

    // The child's facts live in the child's own session.
    let child_checkpoint = coordinator
        .checkpoint(&ConversationId("cc".into()))
        .expect("exports");
    assert_eq!(child_checkpoint.works.len(), 1);
    coordinator.shutdown().await;
}

#[tokio::test]
async fn d4_shutdown_collects_every_held_session() {
    let paused =
        RecordingGateway::scripted(vec![Ok(tooluse_output("p", "echo", serde_json::json!({})))]);
    let done = RecordingGateway::scripted(vec![Ok(endturn_output("done"))]);
    let running_gate = GatedGateway::new("running", true);
    let mut profiles = Profiles::new();
    profiles.insert("paused", Profile::pausing(paused, vec![Arc::new(EchoTool)]));
    profiles.insert("done", Profile::completing(done));
    profiles.insert("running", Profile::completing(running_gate.clone()));
    let coordinator = harness(cfg(4, 2), profiles);

    let paused_request = CreateRequest::root("cp", "kp", parts("p"), "paused");
    let paused_receipt = coordinator
        .create(paused_request.clone())
        .expect("admitted");
    let paused_observation = coordinator
        .wait(&paused_receipt.work, Duration::from_secs(5))
        .await
        .expect("observable");
    assert_eq!(paused_observation.observation.state, WorkState::Paused);

    let done_receipt = coordinator
        .create(CreateRequest::root("cd", "kd", parts("d"), "done"))
        .expect("admitted");
    let _ = coordinator
        .wait(&done_receipt.work, Duration::from_secs(5))
        .await
        .expect("observable");

    // One work is still running when the collective shutdown starts: it must
    // be cancelled and collected, not left behind.
    let running_receipt = coordinator
        .create(CreateRequest::root("cr", "kr", parts("r"), "running"))
        .expect("admitted");
    running_gate.wait_entered().await;

    let paused_handle = coordinator.route(&paused_receipt.work).expect("routed");
    let done_handle = coordinator.route(&done_receipt.work).expect("routed");
    let running_handle = coordinator.route(&running_receipt.work).expect("routed");
    coordinator.shutdown().await;

    // New work is refused...
    assert_eq!(
        coordinator.create(CreateRequest::root("cx", "kx", parts("x"), "done")),
        Err(CoordinatorError::Closed)
    );
    // ...while an accepted request still replays, mirroring the session rule.
    assert_eq!(
        coordinator
            .create(paused_request)
            .expect("a same-key replay survives shutdown"),
        paused_receipt
    );
    // Settled sessions keep their material readable and refuse new work.
    assert_eq!(
        paused_handle
            .observe(&paused_receipt.work)
            .expect("readable")
            .state,
        WorkState::Paused
    );
    assert_eq!(
        paused_handle.submit(session_req("later", "x")),
        Err(SessionError::Closed)
    );
    assert_eq!(
        done_handle.submit(session_req("later-done", "x")),
        Err(SessionError::Closed),
        "the finished session is collected too"
    );
    // The work that was still running when shutdown started was wound down.
    let running_observation = running_handle
        .observe(&running_receipt.work)
        .expect("readable");
    assert!(
        matches!(
            running_observation.state,
            WorkState::Finished | WorkState::Faulted
        ),
        "the collective shutdown settles a running work: {:?}",
        running_observation.state
    );
    assert_eq!(
        running_handle.submit(session_req("later-running", "x")),
        Err(SessionError::Closed)
    );
    assert!(
        coordinator.checkpoint(&ConversationId("cd".into())).is_ok(),
        "a finished session still exports after shutdown"
    );
    assert_eq!(coordinator.session_count(), 3);
}

// ---- review fixes: closed-path, uniqueness, and the remaining D4 clauses ---

#[tokio::test]
async fn d1_a_mismatched_assembly_is_refused() {
    // A harness that assembles a state for another conversation must be
    // refused before anything is registered.
    let coordinator = Arc::new(Coordinator::new(cfg(2, 2), |_request: &CreateRequest| {
        SessionParts {
            state: ConversationState::new(ConversationId("other".into())),
            runner: Arc::new(runner_with(
                RecordingGateway::scripted(vec![Ok(endturn_output("done"))]),
                Vec::new(),
            )),
            options: TurnRunOptions::default(),
            config: SessionConfig::default(),
        }
    }));

    let mismatch = CoordinatorError::ConversationMismatch {
        expected: ConversationId("c1".into()),
        actual: ConversationId("other".into()),
    };
    let error = coordinator
        .create(CreateRequest::root("c1", "k1", parts("x"), "done"))
        .expect_err("a state for another conversation is refused");
    assert_eq!(error, mismatch);
    assert_eq!(coordinator.session_count(), 0, "nothing is registered");
    // The refusal did not consume the key: the same request reaches the same
    // guard again instead of replaying.
    assert_eq!(
        coordinator.create(CreateRequest::root("c1", "k1", parts("x"), "done")),
        Err(mismatch)
    );
}

#[tokio::test]
async fn d4_child_deadline_is_configured_by_the_harness_and_independent() {
    let parent = RecordingGateway::scripted(vec![Ok(endturn_output("parent done"))]);
    let mut profiles = Profiles::new();
    profiles.insert("parent", Profile::completing(parent));
    profiles.insert(
        "child-with-deadline",
        Profile::completing(Arc::new(SlowGateway))
            .with_tools(vec![Arc::new(EchoTool)])
            .with_config(SessionConfig {
                work_deadline: Some(Duration::from_millis(50)),
                ..SessionConfig::default()
            }),
    );
    let coordinator = harness(cfg(2, 2), profiles);

    // The parent runs with no deadline of its own and completes normally.
    let parent_receipt = coordinator
        .create(CreateRequest::root("cp", "kp", parts("p"), "parent"))
        .expect("admitted");
    let parent_done = coordinator
        .wait(&parent_receipt.work, Duration::from_secs(5))
        .await
        .expect("observable");
    assert!(matches!(
        parent_done.observation.finished,
        Some(FinishedKind::Completed { .. })
    ));

    // The child carries the harness-configured per-work deadline (proposal
    // §9: the harness configures it; it is not inherited from the parent or
    // from a wait), so the slow round ends by that deadline.
    let child_receipt = coordinator
        .create(CreateRequest::child(
            "cc",
            "kc",
            parts("c"),
            "child-with-deadline",
            &parent_receipt.work,
        ))
        .expect("admitted");
    let child_done = coordinator
        .wait(&child_receipt.work, Duration::from_secs(5))
        .await
        .expect("observable");
    match child_done.observation.finished {
        Some(FinishedKind::Interrupted { cause, .. }) => assert_eq!(
            cause,
            TurnInterruption::TurnDeadlineExceeded,
            "the child's own work deadline interrupts the slow round"
        ),
        other => panic!("the child deadline must interrupt the slow round, got {other:?}"),
    }

    // The parent's own result is untouched by the child's deadline.
    assert!(matches!(
        coordinator
            .observe(&parent_receipt.work)
            .expect("observable")
            .finished,
        Some(FinishedKind::Completed { .. })
    ));
    coordinator.shutdown().await;
}

#[tokio::test]
async fn d4_a_duplicate_conversation_id_is_refused() {
    let gateway = RecordingGateway::repeating_last(vec![Ok(endturn_output("done"))]);
    let mut profiles = Profiles::new();
    profiles.insert("done", Profile::completing(gateway));
    let coordinator = harness(cfg(2, 2), profiles);

    let first = coordinator
        .create(CreateRequest::root("c1", "k1", parts("one"), "done"))
        .expect("admitted");
    let duplicate = coordinator.create(CreateRequest::root("c1", "k2", parts("two"), "done"));
    assert_eq!(
        duplicate,
        Err(CoordinatorError::ConversationExists(ConversationId(
            "c1".into()
        )))
    );
    assert_eq!(
        coordinator.session_count(),
        1,
        "a create never replaces a live session"
    );

    // The first session is untouched and still routable.
    let waited = coordinator
        .wait(&first.work, Duration::from_secs(5))
        .await
        .expect("observable");
    assert_eq!(waited.observation.state, WorkState::Finished);
    coordinator.shutdown().await;
}

#[tokio::test]
async fn d4_resume_is_refused_once_the_coordinator_is_closed() {
    let gateway =
        RecordingGateway::scripted(vec![Ok(tooluse_output("p", "echo", serde_json::json!({})))]);
    let mut profiles = Profiles::new();
    profiles.insert(
        "paused",
        Profile::pausing(gateway, vec![Arc::new(EchoTool)]),
    );
    let coordinator = harness(cfg(2, 2), profiles);

    let receipt = coordinator
        .create(CreateRequest::root("cp", "kp", parts("p"), "paused"))
        .expect("admitted");
    let paused = coordinator
        .wait(&receipt.work, Duration::from_secs(5))
        .await
        .expect("observable");
    assert_eq!(paused.observation.state, WorkState::Paused);

    coordinator.shutdown().await;

    // A new acceptance cannot slip into a closed coordinator, and the paused
    // material is unchanged.
    assert_eq!(
        coordinator.resume(
            &receipt.work,
            paused.observation.revision,
            "ra",
            approve(vec![awaiting_echo()]),
        ),
        Err(CoordinatorError::Closed)
    );
    let handle = coordinator.route(&receipt.work).expect("routed");
    assert_eq!(
        handle.observe(&receipt.work).expect("readable").state,
        WorkState::Paused
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn d4_creates_racing_a_shutdown_are_either_collected_or_refused() {
    // A gated work keeps the collective shutdown in flight while other
    // creates race it: every attempt is either refused, or admitted and then
    // collected by that same shutdown.
    let gate = GatedGateway::new("running", true);
    let done = RecordingGateway::scripted(vec![Ok(endturn_output("done"))]);
    let mut profiles = Profiles::new();
    profiles.insert("running", Profile::completing(gate.clone()));
    profiles.insert("done", Profile::completing(done));
    let coordinator = harness(cfg(8, 2), profiles);

    let running = coordinator
        .create(CreateRequest::root("cr", "kr", parts("r"), "running"))
        .expect("admitted");
    gate.wait_entered().await;

    let shutdown = {
        let coordinator = Arc::clone(&coordinator);
        tokio::spawn(async move { coordinator.shutdown().await })
    };

    let barrier = Arc::new(Barrier::new(4));
    let mut racers = Vec::new();
    for index in 0..4 {
        let coordinator = Arc::clone(&coordinator);
        let barrier = Arc::clone(&barrier);
        racers.push(tokio::task::spawn_blocking(move || {
            barrier.wait();
            let id = format!("cx{index}");
            let key = format!("kx{index}");
            coordinator
                .create(CreateRequest::root(&id, &key, parts("x"), "done"))
                .map(|receipt| (id, receipt.work))
        }));
    }

    shutdown.await.expect("the shutdown task joins");
    let mut admitted = 0;
    for racer in racers {
        match racer.await.expect("the blocking task joins") {
            Ok((id, work)) => {
                admitted += 1;
                let handle = coordinator
                    .handle(&ConversationId(id.clone()))
                    .expect("an admitted racer is a registered session");
                assert_eq!(
                    handle.submit(session_req("later", "x")),
                    Err(SessionError::Closed),
                    "the same shutdown collects every admitted racer"
                );
                assert!(
                    handle.observe(&work).is_ok(),
                    "its work stays observable after the collective close"
                );
            }
            Err(error) => assert_eq!(
                error,
                CoordinatorError::Closed,
                "a create that loses the race is refused, not silently dropped"
            ),
        }
    }
    assert_eq!(
        coordinator.session_count(),
        1 + admitted,
        "every admitted racer is registered and collected"
    );
    let running_observation = coordinator.observe(&running.work).expect("readable");
    assert!(
        matches!(
            running_observation.state,
            WorkState::Finished | WorkState::Faulted
        ),
        "the running work was wound down: {:?}",
        running_observation.state
    );
}
