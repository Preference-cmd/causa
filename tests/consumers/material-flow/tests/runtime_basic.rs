#![cfg(feature = "runtime")]

use async_trait::async_trait;
use causa::kernel::*;
use causa::runtime::*;
use std::sync::{Arc, Mutex};

struct Recorder(Mutex<Vec<ModelRequest>>);
#[async_trait]
impl ModelGateway for Recorder {
    async fn invoke(
        &self,
        request: &ModelRequest,
        _: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        self.0.lock().unwrap().push(request.clone());
        Ok(ModelOutput {
            response: ModelResponse {
                text: TextPayload::new("answer"),
                tool_calls: vec![],
            },
            stop_reason: ModelStopReason::EndTurn,
            usage: None,
            reasoning: None,
        })
    }
}
fn text(value: &str) -> ContextBlock {
    ContextBlock::new(
        new_block_id(),
        BlockContent::Parts(vec![ContentPart::Text(TextPayload::new(value))]),
        BlockMeta::default(),
    )
}

#[tokio::test]
async fn host_reuses_saved_material_and_explicitly_admits_next_input() {
    let gateway = Arc::new(Recorder(Mutex::new(vec![])));
    let runner = TurnRunner::new(
        gateway.clone(),
        Arc::new(ToolExecutor::new(vec![], ToolExecutorOptions::default()).unwrap()),
    );
    let run = |id, context| {
        runner.run(
            TurnId::new(id),
            context,
            TurnRunOptions::new(ModelRef::new("host-model")),
            RunControl::new(CancellationToken::new(), None),
        )
    };
    let first = run(
        "T1",
        Context::from_blocks(vec![text("first input")]).unwrap(),
    )
    .await;
    assert!(matches!(first.result, TurnResult::Completed { .. }));
    let saved = serde_json::to_vec(&first.context).unwrap();
    let mut retained: Context = serde_json::from_slice(&saved).unwrap();
    let input = text("input admitted by host after T1");
    retained.edit().append([input.clone()]).commit().unwrap();
    let second = run("T2", retained).await;
    assert!(matches!(second.result, TurnResult::Completed { .. }));
    assert_eq!(first.context.blocks().len(), 2);
    assert_eq!(second.context.blocks().len(), 4);
    assert_eq!(second.context.blocks()[2], input);
    let requests = gateway.0.lock().unwrap();
    assert_eq!(requests[1].invocation_id.turn_id, TurnId::new("T2"));
    assert_eq!(requests[1].frame.blocks.len(), 3);
    assert_eq!(&requests[1].frame.blocks[..2], first.context.blocks());
}

struct Summarize {
    summary: ContextBlock,
    request: ContextBlock,
}
#[async_trait]
impl ContextPreparer for Summarize {
    async fn prepare(
        &self,
        context: &mut Context,
        round: &RoundInfo<'_>,
        _: &CallControl,
    ) -> Result<ContextFrame, PrepareError> {
        if round.invocation_id.round_id == RoundId(1) {
            let length = context.blocks().len();
            context
                .edit()
                .replace(0..length, [self.summary.clone(), self.request.clone()])
                .commit()
                .unwrap();
        }
        Ok(context.frame())
    }
}
struct Scripted(Mutex<Vec<ModelRequest>>);
#[async_trait]
impl ModelGateway for Scripted {
    async fn invoke(
        &self,
        request: &ModelRequest,
        _: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        let mut requests = self.0.lock().unwrap();
        let first = requests.is_empty();
        requests.push(request.clone());
        Ok(ModelOutput {
            response: ModelResponse {
                text: TextPayload::new(if first { "" } else { "final answer" }),
                tool_calls: if first {
                    vec![ToolCallDraft {
                        tool_name: "fetch".into(),
                        arguments: serde_json::json!({}),
                        provider_call_id: Some("call-fetch".into()),
                    }]
                } else {
                    vec![]
                },
            },
            stop_reason: if first {
                ModelStopReason::ToolUse
            } else {
                ModelStopReason::EndTurn
            },
            usage: None,
            reasoning: None,
        })
    }
}
struct Fetch(std::sync::atomic::AtomicUsize);
#[async_trait]
impl Tool for Fetch {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "fetch".into(),
            description: "fetch material".into(),
            parameters: serde_json::json!({"type":"object"}),
        }
    }
    async fn execute(&self, call: &ToolCallContext, _: &CallControl) -> ToolResultPayload {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        ToolResultPayload {
            call_block_id: call.call_block_id,
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(serde_json::json!("fetched content")),
            media: vec![],
            notes: vec![],
        }
    }
}
#[tokio::test]
async fn summary_removes_old_material_across_rounds_and_runs_without_rewriting_saved_observations()
{
    let a = text("A original research");
    let b = text("B original research");
    let input = text("U current request");
    let summary = text("S research summary");
    let initial = Context::from_blocks(vec![a.clone(), b.clone(), input.clone()]).unwrap();
    let saved_context = initial.clone();
    let saved_frame = initial.frame();
    let appended = Arc::new(Mutex::new(vec![]));
    let saved = appended.clone();
    let gateway = Arc::new(Scripted(Mutex::new(vec![])));
    let tool = Arc::new(Fetch(std::sync::atomic::AtomicUsize::new(0)));
    let runner = TurnRunner::new(
        gateway.clone(),
        Arc::new(ToolExecutor::new(vec![tool.clone()], ToolExecutorOptions::default()).unwrap()),
    );
    let mut options = TurnRunOptions::new(ModelRef::new("host-model"));
    options.preparer = Some(Arc::new(Summarize {
        summary: summary.clone(),
        request: input.clone(),
    }));
    options.observer = Some(Arc::new(move |event| {
        if let RunEvent::BlocksCommitted { blocks, .. } = event {
            saved.lock().unwrap().extend_from_slice(blocks);
        }
    }));
    let first = runner
        .run(
            TurnId::new("T1"),
            initial,
            options,
            RunControl::new(CancellationToken::new(), None),
        )
        .await;
    assert!(matches!(first.result, TurnResult::Completed { .. }));
    assert_eq!(
        &first.context.blocks()[..2],
        &[summary.clone(), input.clone()]
    );
    assert_eq!(first.context.blocks().len(), 3);
    assert!(
        first
            .context
            .blocks()
            .iter()
            .all(|block| matches!(block.content(), BlockContent::Parts(_)))
    );
    let mut second_context = first.context;
    let next_input = text("U2 next request");
    second_context
        .edit()
        .append([next_input.clone()])
        .commit()
        .unwrap();
    let second = runner
        .run(
            TurnId::new("T2"),
            second_context,
            TurnRunOptions::new(ModelRef::new("host-model")),
            RunControl::new(CancellationToken::new(), None),
        )
        .await;
    assert!(matches!(second.result, TurnResult::Completed { .. }));
    assert_eq!(tool.0.load(std::sync::atomic::Ordering::SeqCst), 1);
    let requests = gateway.0.lock().unwrap();
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[0].frame.blocks, saved_frame.blocks);
    assert_eq!(requests[1].frame.blocks, vec![summary.clone(), input]);
    assert_eq!(requests[2].frame.blocks.len(), 4);
    assert_eq!(requests[2].frame.blocks[3], next_input);
    for request in &requests[1..] {
        assert!(
            request
                .frame
                .blocks
                .iter()
                .all(|block| block.id() != a.id() && block.id() != b.id())
        );
    }
    assert_eq!(saved_context.blocks(), saved_frame.blocks);
    let observations = appended.lock().unwrap();
    let old_call = observations
        .iter()
        .find(|block| matches!(block.content(), BlockContent::ToolCall(_)))
        .unwrap();
    assert!(observations.iter().any(|block| matches!(block.content(), BlockContent::ToolResult(result) if result.call_block_id == old_call.id())));
    assert!(
        second
            .context
            .blocks()
            .iter()
            .all(|block| block.id() != old_call.id())
    );
    assert!(!observations.iter().any(|block| block.id() == summary.id())); // preparation does not emit runner commits
    assert_ne!(summary.id(), a.id());
    assert_ne!(summary.id(), b.id());
}
