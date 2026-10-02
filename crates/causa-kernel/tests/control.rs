//! Absolute parent deadlines and local control remain composable.

use causa_kernel::{CallControl, CancellationToken, ControlError};
use std::time::{Duration, Instant};

#[test]
fn timeout_narrows_absolute_deadline_and_clone_keeps_clock() {
    let token = CancellationToken::new();
    let deadline = Instant::now() + Duration::from_secs(30);
    let parent = CallControl::new(token, Some(deadline));
    assert_eq!(parent.deadline(), Some(deadline));
    assert_eq!(parent.clone().deadline(), Some(deadline));
    assert_eq!(
        parent.with_timeout(Duration::from_secs(60)).deadline(),
        Some(deadline)
    );
    let start = Instant::now();
    let child = parent.with_timeout(Duration::from_secs(1));
    assert!(child.deadline().unwrap() >= start + Duration::from_secs(1));
    assert!(child.deadline().unwrap() < deadline);
    assert_eq!(parent.deadline(), Some(deadline));
}

#[test]
fn local_expiry_never_cancels_parent_and_parent_cancellation_is_inherited() {
    let token = CancellationToken::new();
    let parent = CallControl::new(token.clone(), None);
    let child = parent.with_timeout(Duration::ZERO);
    assert!(matches!(child.check(), Err(ControlError::TimedOut)));
    assert!(!parent.is_cancelled());
    assert!(parent.check().is_ok());
    token.cancel();
    assert!(matches!(parent.check(), Err(ControlError::Cancelled)));
    assert!(matches!(child.check(), Err(ControlError::Cancelled)));
}

#[test]
fn extreme_timeout_does_not_panic_or_extend_parent() {
    let deadline = Instant::now() + Duration::from_secs(1);
    let parent = CallControl::new(CancellationToken::new(), Some(deadline));
    assert_eq!(
        parent.with_timeout(Duration::MAX).deadline(),
        Some(deadline)
    );
    let unbounded = CallControl::new(CancellationToken::new(), None);
    assert_eq!(unbounded.with_timeout(Duration::MAX).deadline(), None);
}

#[test]
fn absolute_expired_deadline_is_preserved_by_all_views() {
    let deadline = Instant::now() - Duration::from_secs(1);
    let parent = CallControl::new(CancellationToken::new(), Some(deadline));
    assert!(matches!(parent.check(), Err(ControlError::TimedOut)));
    assert_eq!(
        parent.with_timeout(Duration::from_secs(60)).deadline(),
        Some(deadline)
    );
    assert!(matches!(
        parent.clone().check(),
        Err(ControlError::TimedOut)
    ));
}
