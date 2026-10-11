//! Cancellation and deadline wait helper for runtime operations.

use causa_kernel::{CallControl, ControlError};

/// Races the control plane's cancellation token and optional deadline.
pub(crate) async fn wait_for_stop(control: &CallControl) -> ControlError {
    tokio::select! {
        biased;
        _ = control.cancellation_token().cancelled() => ControlError::Cancelled,
        _ = async {
            match control.deadline() {
                Some(deadline) => tokio::time::sleep_until(deadline.into()).await,
                None => std::future::pending::<()>().await,
            }
        } => ControlError::TimedOut,
    }
}
