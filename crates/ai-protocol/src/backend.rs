//! Adapter-internal completion seam.
//!
//! The concrete adapters (OpenAI-compatible, Anthropic) implement
//! `AgentProvider` by delegating network calls to a `CompletionBackend`.
//! Tests use `FakeCompletionBackend` to script behavior without live
//! network. The real reqwest-backed backend lives in `reqwest_backend.rs`.
//!
//! `CompletionBackend` is intentionally NOT exported as the
//! Reimagine provider abstraction — that role belongs to
//! `reimagine_agent_harness::AgentProvider`.
//!
//! Implementation note: the plan sketched a custom `PopFront` trait to
//! give `Vec` a `pop_front` method. We deviate by using
//! `std::collections::VecDeque` directly; it provides both `pop_front`
//! and `push_front` (used to put a non-matching step back) without
//! the extra trait plumbing.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use reimagine_agent_harness::{
    AgentRequest, AgentResponse, AgentStream, AgentStreamEvent, ModelInfo,
};

use crate::error::ProviderAdapterError;

/// One step the fake backend will replay. `Complete` and `Stream` are
/// separate so a single test can mix completion and streaming flows.
#[derive(Debug, Clone)]
pub enum ScriptedBackendStep {
    /// A canned `complete` call result.
    Complete(Result<AgentResponse, ProviderAdapterError>),
    /// A canned `stream` call result: a sequence of stream events.
    /// An empty vec means "exhausted immediately".
    Stream(Vec<Result<AgentStreamEvent, ProviderAdapterError>>),
}

/// The internal backend seam. Implementations translate Reimagine
/// provider-agnostic shapes to whatever transport the upstream uses.
#[async_trait]
pub trait CompletionBackend: Send + Sync {
    async fn complete(&self, request: AgentRequest) -> Result<AgentResponse, ProviderAdapterError>;

    /// Return a boxed stream of events. The `Result` outer layer is
    /// for setup-time errors (e.g. transport); inner items are
    /// per-event errors that the stream itself surfaces.
    async fn stream(
        &self,
        request: AgentRequest,
    ) -> Result<Box<dyn AgentStream>, ProviderAdapterError>;

    async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderAdapterError>;
}

/// In-memory scripted backend. Used by tests. The fake consumes
/// scripted steps in order; if a request has no remaining step, the
/// call returns `ProviderAdapterError::configuration("scripted backend
/// exhausted")`. The fake also keeps a static `models` list returned
/// by `list_models`.
#[derive(Debug, Default)]
pub struct FakeCompletionBackend {
    steps: Arc<Mutex<VecDeque<ScriptedBackendStep>>>,
    models: Vec<ModelInfo>,
}

impl FakeCompletionBackend {
    pub fn new(steps: Vec<ScriptedBackendStep>) -> Self {
        Self {
            steps: Arc::new(Mutex::new(steps.into_iter().collect())),
            models: Vec::new(),
        }
    }

    pub fn with_models(mut self, models: Vec<ModelInfo>) -> Self {
        self.models = models;
        self
    }
}

#[async_trait]
impl CompletionBackend for FakeCompletionBackend {
    async fn complete(
        &self,
        _request: AgentRequest,
    ) -> Result<AgentResponse, ProviderAdapterError> {
        let mut steps = self.steps.lock().unwrap();
        match steps.pop_front() {
            Some(ScriptedBackendStep::Complete(r)) => r,
            Some(other) => {
                // Put it back so a later `stream` call can consume it.
                steps.push_front(other);
                Err(ProviderAdapterError::configuration(
                    "scripted backend next step is not a Complete step",
                ))
            }
            None => Err(ProviderAdapterError::configuration(
                "scripted backend exhausted",
            )),
        }
    }

    async fn stream(
        &self,
        _request: AgentRequest,
    ) -> Result<Box<dyn AgentStream>, ProviderAdapterError> {
        let events = {
            let mut steps = self.steps.lock().unwrap();
            match steps.pop_front() {
                Some(ScriptedBackendStep::Stream(events)) => events,
                Some(other) => {
                    steps.push_front(other);
                    return Err(ProviderAdapterError::configuration(
                        "scripted backend next step is not a Stream step",
                    ));
                }
                None => {
                    return Err(ProviderAdapterError::configuration(
                        "scripted backend exhausted",
                    ));
                }
            }
        };
        Ok(Box::new(ScriptedStream { events }))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderAdapterError> {
        Ok(self.models.clone())
    }
}

/// Tiny in-memory stream that yields the scripted events once, then
/// returns `None`.
struct ScriptedStream {
    events: Vec<Result<AgentStreamEvent, ProviderAdapterError>>,
}

#[async_trait]
impl AgentStream for ScriptedStream {
    async fn next_event(&mut self) -> Option<AgentStreamEvent> {
        if self.events.is_empty() {
            return None;
        }
        // We deliberately drop per-event errors at the fake boundary:
        // scripted `Result` only carries the event itself, errors are
        // signalled by returning `None` to the runtime.
        self.events.remove(0).ok()
    }
}
