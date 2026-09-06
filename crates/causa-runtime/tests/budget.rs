//! Reference budget tests — the frame-materialization policy moved here
//! from the kernel (Slice 13): it is the reference harness's opinion over
//! the kernel's lossless projection, so its contract rides with the
//! runtime. What stays in the kernel is the fact-machine side: the
//! lossless `TurnContext::frame` is a pure function of the committed facts.

mod common;

use common::{DropAllCompaction, ctx};

use causa_kernel::{RoundId, TextPayload};
use causa_runtime::{FramePolicy, TokenCounter, WindowBudget};

/// A host-chosen estimator: any non-empty content trips the trigger.
/// (The default policy has no counter and estimates zero — compaction is
/// genuinely opt-in via a wired `TokenCounter`.)
struct CountPlusOne;
impl TokenCounter for CountPlusOne {
    fn estimate(&self, _blocks: &[causa_kernel::ContextBlock]) -> usize {
        101
    }
    fn estimate_value(&self, _value: &serde_json::Value) -> usize {
        1
    }
}

/// Compaction output is frame-local: the projected frame keeps the
/// deterministic frame identity of the lossless projection, only the block
/// list is replaced, and the fact state is never written back.
#[tokio::test]
async fn compaction_projection_identity() {
    let mut c = ctx("t1");
    c.append_input(TextPayload::new("hello"), "user").unwrap();
    let lossless = FramePolicy::default();
    let compacting = FramePolicy {
        window_budget: WindowBudget {
            model_window_limit: 100,
            compaction_trigger: 1,
        },
        compaction: Some(std::sync::Arc::new(DropAllCompaction)),
        token_counter: Some(std::sync::Arc::new(CountPlusOne)),
    };
    let sync_frame = c.frame(RoundId(0));
    let lossless_frame = lossless.materialize(&c, RoundId(0)).await.unwrap();
    assert_eq!(sync_frame.frame_id, lossless_frame.frame_id);
    assert_eq!(
        serde_json::to_string(&sync_frame.model_context.blocks).unwrap(),
        serde_json::to_string(&lossless_frame.model_context.blocks).unwrap()
    );
    assert_eq!(
        serde_json::to_string(&lossless_frame.model_context.blocks).unwrap(),
        serde_json::to_string(&c.snapshot_blocks()).unwrap()
    );
    let projected = compacting.materialize(&c, RoundId(0)).await.unwrap();
    assert_eq!(projected.frame_id, sync_frame.frame_id);
    assert!(projected.model_context.blocks.is_empty());
    assert_eq!(c.snapshot_blocks().len(), 1);
}
