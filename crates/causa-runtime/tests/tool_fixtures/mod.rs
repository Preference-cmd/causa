#![allow(dead_code)]

use async_trait::async_trait;
use causa_kernel::*;
use causa_runtime::{ToolExecutor, ToolExecutorOptions, new_block_id};
use serde_json::json;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};

pub fn invocation() -> InvocationId {
    InvocationId {
        turn_id: TurnId::new("standalone"),
        round_id: RoundId(3),
    }
}
pub fn control() -> CallControl {
    CallControl::new(CancellationToken::new(), None)
}
pub fn definition(name: &str) -> ToolDefinition {
    ToolDefinition {
        name: name.into(),
        description: format!("definition for {name}"),
        parameters: json!({"type": "object"}),
    }
}
pub fn call(name: &str) -> ToolCallContext {
    ToolCallContext::from_declaration(
        new_block_id(),
        &ToolCallPayload {
            tool_name: name.into(),
            arguments: json!({"q": 7}),
        },
    )
}
pub fn batch(names: &[&str]) -> ToolBatch {
    ToolBatch::new(names.iter().map(|name| call(name)).collect()).unwrap()
}
pub fn result(
    call: &ToolCallContext,
    status: ToolResultStatus,
    content: serde_json::Value,
) -> ToolResultPayload {
    ToolResultPayload {
        call_block_id: call.call_block_id,
        status,
        output: ToolOutput::new(content),
        media: vec![],
        notes: vec![],
    }
}
pub fn executor(tools: Vec<Arc<dyn Tool>>, options: ToolExecutorOptions) -> ToolExecutor {
    ToolExecutor::new(tools, options).unwrap()
}
pub fn statuses(batch: &ToolBatch) -> Vec<ToolResultStatus> {
    batch
        .results()
        .iter()
        .map(|entry| entry.result().unwrap().1.status.clone())
        .collect()
}
pub struct NamedTool {
    pub name: Mutex<String>,
    pub label: &'static str,
    pub count: AtomicUsize,
}
impl NamedTool {
    pub fn new(name: &str) -> Arc<Self> {
        Self::labeled(name, "static")
    }
    pub fn labeled(name: &str, label: &'static str) -> Arc<Self> {
        Arc::new(Self {
            name: Mutex::new(name.into()),
            label,
            count: AtomicUsize::new(0),
        })
    }
}
#[async_trait]
impl Tool for NamedTool {
    fn definition(&self) -> ToolDefinition {
        definition(&self.name.lock().unwrap())
    }
    async fn execute(&self, call: &ToolCallContext, _control: &CallControl) -> ToolResultPayload {
        self.count.fetch_add(1, Ordering::SeqCst);
        let mut result = result(
            call,
            ToolResultStatus::Succeeded,
            json!({"object": self.label, "arguments": call.input.arguments}),
        );
        result.notes.push(TextPayload::new("tool note"));
        result
    }
}
pub struct Source {
    pub id: &'static str,
    pub label: &'static str,
    pub definitions: Mutex<Vec<ToolDefinition>>,
    pub version: AtomicU64,
    pub lists: AtomicUsize,
    pub invocations: AtomicUsize,
    pub error: Mutex<Option<SourceError>>,
}
impl Source {
    pub fn new(id: &'static str, label: &'static str, names: &[&str]) -> Arc<Self> {
        Arc::new(Self {
            id,
            label,
            definitions: Mutex::new(names.iter().map(|name| definition(name)).collect()),
            version: AtomicU64::new(0),
            lists: AtomicUsize::new(0),
            invocations: AtomicUsize::new(0),
            error: Mutex::new(None),
        })
    }
}
#[async_trait]
impl DynamicToolSource for Source {
    fn id(&self) -> &str {
        self.id
    }
    fn version(&self) -> u64 {
        self.version.load(Ordering::SeqCst)
    }
    async fn list(&self) -> Result<Vec<ToolDefinition>, SourceError> {
        self.lists.fetch_add(1, Ordering::SeqCst);
        if let Some(error) = self.error.lock().unwrap().clone() {
            return Err(error);
        }
        Ok(self.definitions.lock().unwrap().clone())
    }
    async fn invoke(
        &self,
        call: &ToolCallContext,
        _control: &CallControl,
    ) -> Result<ToolResultPayload, ToolExecutionError> {
        self.invocations.fetch_add(1, Ordering::SeqCst);
        Ok(result(
            call,
            ToolResultStatus::Succeeded,
            json!({"object": self.label, "arguments": call.input.arguments}),
        ))
    }
}
