//! The session owner — [`Session`] (the non-`Clone` lifecycle anchor) and the
//! by-value construction rejection it returns.
//!
//! The owner holds the private execution core behind an `Arc`; cloneable
//! [`SessionHandle`]s are minted from it and share the same core.

use std::sync::Arc;

use crate::config::TurnRunOptions;
use crate::conversation::ConversationState;
use crate::driver::TurnRunner;

use super::execution::SessionCore;
use super::handle::SessionHandle;
use super::types::{SessionConfig, SessionError};

/// Returned by [`Session::new`] when the supplied state is not idle.
///
/// A by-value constructor that rejects hands back **all** the inputs it
/// consumed, not only the state: the caller keeps its [`ConversationState`], its
/// assembled `runner` and `options`, and its `config` for a later attempt.
/// Every field is cheap to move back — `runner` is an `Arc`, `options` and
/// `config` are plain values.
pub struct SessionBuildRejection {
    /// Why construction was rejected.
    pub error: SessionError,
    /// The state, returned by value.
    pub state: ConversationState,
    /// The assembled runner, returned by value.
    pub runner: Arc<TurnRunner>,
    /// The assembled run options, returned by value.
    pub options: TurnRunOptions,
    /// The session configuration, returned by value.
    pub config: SessionConfig,
}

impl std::fmt::Debug for SessionBuildRejection {
    /// `TurnRunner` carries no `Debug` impl (it holds gateway / executor trait
    /// objects), so the rejection renders it as an opaque handle and prints
    /// the rest.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionBuildRejection")
            .field("error", &self.error)
            .field("state", &self.state)
            .field("runner", &"Arc<TurnRunner>")
            .field("options", &self.options)
            .field("config", &self.config)
            .finish()
    }
}

/// The session owner: one per conversation, held by the harness.
///
/// Non-`Clone` by design — the owner is the lifecycle anchor. Cloneable
/// [`SessionHandle`]s are the operation entries; they neither own the
/// lifecycle nor expose the writable state.
pub struct Session {
    core: Arc<SessionCore>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("conversation_id", &self.core.conversation_id())
            .finish_non_exhaustive()
    }
}

impl Session {
    /// Build an idle session from a validated [`ConversationState`] plus the
    /// harness's assembled runner and options.
    ///
    /// Construction never calls the model or a tool and never starts work; a
    /// work is admitted only by `submit`. A state with an active turn (or, by
    /// the aggregate's invariant, a leftover stamp) is rejected, and every
    /// by-value input (`state`, `runner`, `options`, `config`) comes back in
    /// [`SessionBuildRejection`].
    ///
    /// # Errors
    ///
    /// Returns [`SessionBuildRejection`] carrying
    /// [`SessionError::InvalidInput`] when `state` is not idle.
    // The rejection carries the whole `ConversationState` plus the assembled
    // runner / options / config back by value, which is the point of the
    // contract — boxing it would only move the inputs behind another
    // allocation.
    #[allow(clippy::result_large_err)]
    pub fn new(
        state: ConversationState,
        runner: Arc<TurnRunner>,
        options: TurnRunOptions,
        config: SessionConfig,
    ) -> Result<Session, SessionBuildRejection> {
        if state.active_turn().is_some() || state.sealed_result().is_some() {
            return Err(SessionBuildRejection {
                error: SessionError::InvalidInput(
                    "session construction requires an idle state (empty active slot)".into(),
                ),
                state,
                runner,
                options,
                config,
            });
        }
        Ok(Session {
            core: SessionCore::new(state, runner, options, config),
        })
    }

    /// A cloneable operation entry into this session.
    pub fn handle(&self) -> SessionHandle {
        SessionHandle {
            core: Arc::clone(&self.core),
        }
    }
}

impl Drop for Session {
    /// Stops acceptance and sends the stop signal to the active work.
    ///
    /// The worker keeps its own `Arc<SessionCore>`, so no strong reference
    /// cycle exists and the in-flight run still publishes its terminal state
    /// after the owner is gone. Drop cannot await that wind-down, nor persist
    /// anything — an explicit `shutdown` covers those.
    fn drop(&mut self) {
        self.core.close();
    }
}
