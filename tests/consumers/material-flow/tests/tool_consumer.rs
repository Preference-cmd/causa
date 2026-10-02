#![cfg(feature = "runtime")]

use async_trait::async_trait;
use causa::kernel::*;
use causa::runtime::*;
use serde_json::json;
use std::sync::Arc;

struct LocalTool;
#[async_trait]
impl Tool for LocalTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "local".into(),
            description: "Host-owned tool".into(),
            parameters: json!({"type":"object"}),
        }
    }
    async fn execute(&self, call: &ToolCallContext, _: &CallControl) -> ToolResultPayload {
        ToolResultPayload {
            call_block_id: call.call_block_id,
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(call.input.arguments.clone()),
            media: vec![],
            notes: vec![],
        }
    }
}
#[tokio::test]
async fn standalone_host_binds_processes_and_decides_when_to_commit() {
    let executor =
        ToolExecutor::new(vec![Arc::new(LocalTool)], ToolExecutorOptions::default()).unwrap();
    let invocation = InvocationId {
        turn_id: TurnId::new("standalone"),
        round_id: RoundId(3),
    };
    let bound = executor
        .bind(invocation, CallControl::new(CancellationToken::new(), None))
        .await
        .unwrap();
    assert_eq!(bound.surface().definitions[0].name, "local");
    let declaration = ContextBlock::new(
        new_block_id(),
        BlockContent::ToolCall(ToolCallPayload {
            tool_name: "local".into(),
            arguments: json!({"host":true}),
        }),
        BlockMeta::default(),
    );
    let BlockContent::ToolCall(payload) = declaration.content() else {
        unreachable!()
    };
    let mut batch = ToolBatch::new(vec![ToolCallContext::from_declaration(
        declaration.id(),
        payload,
    )])
    .unwrap();
    let mut context = Context::from_blocks(vec![declaration]).unwrap();
    bound.process(&mut batch).await.unwrap();
    assert_eq!(context.blocks().len(), 1);
    let results = batch
        .results()
        .iter()
        .map(|entry| {
            let (id, result) = entry.result().unwrap();
            (*id, result.clone())
        })
        .collect::<Vec<_>>();
    validate_tool_result_append(context.blocks(), &results).unwrap();
    assert_eq!(results[0].1.output.content, json!({"host":true}));
    context
        .edit()
        .append(results.into_iter().map(|(id, result)| {
            ContextBlock::new(id, BlockContent::ToolResult(result), BlockMeta::default())
        }))
        .commit()
        .unwrap();
    assert_eq!(context.blocks().len(), 2);
}
