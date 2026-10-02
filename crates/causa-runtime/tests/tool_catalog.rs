//! Deterministic refresh races prove private cache and registration ownership.
mod tool_fixtures;
use async_trait::async_trait;
use causa_kernel::*;
use causa_runtime::{ToolCatalogError, ToolExecutorOptions};
use std::sync::{
    Arc,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};
use tokio::sync::{Notify, Semaphore};
use tool_fixtures::*;

struct RacingSource {
    version: AtomicU64,
    lists: AtomicUsize,
    entered: Arc<Notify>,
    release: Arc<Semaphore>,
}
impl RacingSource {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            version: AtomicU64::new(0),
            lists: AtomicUsize::new(0),
            entered: Arc::new(Notify::new()),
            release: Arc::new(Semaphore::new(0)),
        })
    }
}
#[async_trait]
impl DynamicToolSource for RacingSource {
    fn id(&self) -> &str {
        "racing"
    }
    fn version(&self) -> u64 {
        self.version.load(Ordering::SeqCst)
    }
    async fn list(&self) -> Result<Vec<ToolDefinition>, SourceError> {
        let index = self.lists.fetch_add(1, Ordering::SeqCst);
        if index == 0 {
            self.entered.notify_one();
            self.release.acquire().await.unwrap().forget();
            Ok(vec![definition("old")])
        } else {
            Ok(vec![definition("new")])
        }
    }
    async fn invoke(
        &self,
        call: &ToolCallContext,
        _control: &CallControl,
    ) -> Result<ToolResultPayload, ToolExecutionError> {
        Ok(result(
            call,
            ToolResultStatus::Succeeded,
            serde_json::json!(call.input.tool_name),
        ))
    }
}

#[tokio::test]
async fn cache_hit_avoids_list_and_failed_refresh_keeps_original_source_error() {
    let source = Source::new("source", "source", &["old"]);
    let executor = executor(vec![], ToolExecutorOptions::default());
    executor.register_dynamic(source.clone()).unwrap();
    executor.bind(invocation(), control()).await.unwrap();
    executor.bind(invocation(), control()).await.unwrap();
    assert_eq!(source.lists.load(Ordering::SeqCst), 1);
    source.version.store(1, Ordering::SeqCst);
    *source.error.lock().unwrap() = Some(SourceError::Protocol("original listing failure".into()));
    assert!(
        matches!(executor.bind(invocation(), control()).await, Err(ToolCatalogError::Source { source_id, error: SourceError::Protocol(message) }) if source_id == "source" && message == "original listing failure")
    );
    assert_eq!(source.lists.load(Ordering::SeqCst), 2);
    assert_eq!(source.invocations.load(Ordering::SeqCst), 0);
    *source.error.lock().unwrap() = None;
    *source.definitions.lock().unwrap() = vec![definition("new")];
    assert_eq!(
        executor
            .bind(invocation(), control())
            .await
            .unwrap()
            .surface()
            .definitions[0]
            .name,
        "new"
    );
    assert_eq!(source.lists.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn empty_listing_is_success_and_first_listing_error_is_not_an_empty_catalog() {
    let source = Source::new("source", "source", &[]);
    let executor = executor(vec![], ToolExecutorOptions::default());
    executor.register_dynamic(source.clone()).unwrap();
    assert!(
        executor
            .bind(invocation(), control())
            .await
            .unwrap()
            .surface()
            .definitions
            .is_empty()
    );
    source.version.store(1, Ordering::SeqCst);
    *source.error.lock().unwrap() = Some(SourceError::Unavailable("offline".into()));
    assert!(
        matches!(executor.bind(invocation(), control()).await, Err(ToolCatalogError::Source { error: SourceError::Unavailable(message), .. }) if message == "offline")
    );
}

#[tokio::test]
async fn older_same_version_refresh_cannot_overwrite_newer_concurrent_refresh() {
    let source = RacingSource::new();
    let executor = Arc::new(executor(vec![], ToolExecutorOptions::default()));
    executor.register_dynamic(source.clone()).unwrap();
    let old_executor = executor.clone();
    let old = tokio::spawn(async move {
        let bound = old_executor.bind(invocation(), control()).await.unwrap();
        let surface = bound.surface().definitions.clone();
        let mut calls = batch(&["old"]);
        bound.process(&mut calls).await.unwrap();
        (surface, calls)
    });
    tokio::time::timeout(Duration::from_secs(2), source.entered.notified())
        .await
        .unwrap();
    let new = executor.bind(invocation(), control()).await.unwrap();
    assert_eq!(new.surface().definitions[0].name, "new");
    source.release.add_permits(1);
    let (surface, calls) = tokio::time::timeout(Duration::from_secs(2), old)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(surface[0].name, "old");
    assert_eq!(
        calls.results()[0].result().unwrap().1.output.content,
        serde_json::json!("old")
    );
    assert_eq!(
        executor
            .bind(invocation(), control())
            .await
            .unwrap()
            .surface()
            .definitions[0]
            .name,
        "new"
    );
    assert_eq!(source.lists.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn version_change_during_listing_is_usable_once_and_never_cached_as_current() {
    let source = RacingSource::new();
    let executor = Arc::new(executor(vec![], ToolExecutorOptions::default()));
    executor.register_dynamic(source.clone()).unwrap();
    let binding_executor = executor.clone();
    let first = tokio::spawn(async move {
        binding_executor
            .bind(invocation(), control())
            .await
            .unwrap()
            .surface()
            .definitions
            .clone()
    });
    tokio::time::timeout(Duration::from_secs(2), source.entered.notified())
        .await
        .unwrap();
    source.version.store(1, Ordering::SeqCst);
    source.release.add_permits(1);
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), first)
            .await
            .unwrap()
            .unwrap()[0]
            .name,
        "old"
    );
    assert_eq!(source.lists.load(Ordering::SeqCst), 1);
    assert_eq!(
        executor
            .bind(invocation(), control())
            .await
            .unwrap()
            .surface()
            .definitions[0]
            .name,
        "new"
    );
    assert_eq!(source.lists.load(Ordering::SeqCst), 2);
    executor.bind(invocation(), control()).await.unwrap();
    assert_eq!(source.lists.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn same_arc_reregistration_isolates_delayed_old_list_from_new_registration_cache() {
    let source = RacingSource::new();
    let executor = Arc::new(executor(vec![], ToolExecutorOptions::default()));
    executor.register_dynamic(source.clone()).unwrap();
    let old_executor = executor.clone();
    let old = tokio::spawn(async move {
        old_executor
            .bind(invocation(), control())
            .await
            .unwrap()
            .surface()
            .definitions
            .clone()
    });
    tokio::time::timeout(Duration::from_secs(2), source.entered.notified())
        .await
        .unwrap();
    let returned = executor.unregister_dynamic("racing").unwrap();
    let erased: Arc<dyn DynamicToolSource> = source.clone();
    assert!(Arc::ptr_eq(&returned, &erased));
    executor.register_dynamic(returned).unwrap();
    assert_eq!(
        executor
            .bind(invocation(), control())
            .await
            .unwrap()
            .surface()
            .definitions[0]
            .name,
        "new"
    );
    source.release.add_permits(1);
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), old)
            .await
            .unwrap()
            .unwrap()[0]
            .name,
        "old"
    );
    assert_eq!(
        executor
            .bind(invocation(), control())
            .await
            .unwrap()
            .surface()
            .definitions[0]
            .name,
        "new"
    );
    assert_eq!(source.lists.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn directory_cancel_and_deadline_stop_uncooperative_listing() {
    for deadline in [false, true] {
        let source = RacingSource::new();
        let executor = Arc::new(executor(vec![], ToolExecutorOptions::default()));
        executor.register_dynamic(source.clone()).unwrap();
        let parent = CallControl::new(
            CancellationToken::new(),
            deadline.then(|| Instant::now() + Duration::from_millis(25)),
        );
        let caller = parent.clone();
        let task = tokio::spawn(async move {
            executor
                .bind(invocation(), parent)
                .await
                .map(|bound| bound.surface().definitions.clone())
        });
        tokio::time::timeout(Duration::from_secs(2), source.entered.notified())
            .await
            .unwrap();
        if !deadline {
            caller.cancellation_token().cancel();
        }
        let error = tokio::time::timeout(Duration::from_secs(2), task)
            .await
            .unwrap()
            .unwrap()
            .unwrap_err();
        assert!(matches!(
            (deadline, error),
            (false, ToolCatalogError::Control(ControlError::Cancelled))
                | (true, ToolCatalogError::Control(ControlError::TimedOut))
        ));
        if deadline {
            assert!(!caller.is_cancelled());
        }
        assert_eq!(source.lists.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn maximum_version_is_a_valid_cache_key_without_a_sentinel_collision() {
    let source = Source::new("source", "source", &["tool"]);
    source.version.store(u64::MAX, Ordering::SeqCst);
    let executor = executor(vec![], ToolExecutorOptions::default());
    executor.register_dynamic(source.clone()).unwrap();
    executor.bind(invocation(), control()).await.unwrap();
    executor.bind(invocation(), control()).await.unwrap();
    assert_eq!(source.lists.load(Ordering::SeqCst), 1);
}

struct CancelThenFail(CancellationToken);
#[async_trait]
impl DynamicToolSource for CancelThenFail {
    fn id(&self) -> &str {
        "cancel"
    }
    async fn list(&self) -> Result<Vec<ToolDefinition>, SourceError> {
        self.0.cancel();
        Err(SourceError::Unavailable("simultaneous failure".into()))
    }
    async fn invoke(
        &self,
        _call: &ToolCallContext,
        _control: &CallControl,
    ) -> Result<ToolResultPayload, ToolExecutionError> {
        unreachable!()
    }
}
#[tokio::test]
async fn listing_that_synchronously_cancels_then_fails_obeys_parent_control_priority() {
    let parent = control();
    let executor = executor(vec![], ToolExecutorOptions::default());
    executor
        .register_dynamic(Arc::new(CancelThenFail(
            parent.cancellation_token().clone(),
        )))
        .unwrap();
    assert!(matches!(
        executor.bind(invocation(), parent).await,
        Err(ToolCatalogError::Control(ControlError::Cancelled))
    ));
}
