// The "Media" guide. Runs offline: a tool draws a chart and stores its
// bytes, the user adds a PDF, and the same context is sent through the
// three gateways to a local server. The program prints the parts of each
// request.

// ANCHOR: imports
use async_trait::async_trait;
use base64::Engine;
use causa::{
    kernel::{
        ArtifactHint, ArtifactKind, ArtifactRef, ArtifactStore, BlockContent, BlockMeta,
        CallControl, CancellationToken, ContentPart, Context, ContextBlock, MediaRef, ModelRef,
        Role, StoreError, TextPayload, Tool, ToolCallContext, ToolDefinition, ToolOutput,
        ToolResultPayload, ToolResultStatus, TurnId,
    },
    providers::{MediaPayload, MediaResolver},
    runtime::{ToolExecutor, ToolExecutorOptions, TurnRunOptions, TurnRunner, new_block_id},
};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
// ANCHOR_END: imports

use causa::kernel::{GenerationOptions, ModelGateway};
use causa::providers::{
    AnthropicMessagesGateway, OpenAiChatCompletionsGateway, OpenAiResponsesGateway,
};
use causa_site_guides::fake_provider::FakeProvider;
use causa_site_guides::scripted::{ScriptedModel, call, reply, tool_results};
use serde_json::{Value, json};

// ANCHOR: assets
/// The host's asset table: bytes by content hash. Tools write to it as an
/// `ArtifactStore`; gateways read from it as a `MediaResolver`.
#[derive(Default)]
struct Assets(Mutex<HashMap<String, Vec<u8>>>);

#[async_trait]
impl ArtifactStore for Assets {
    async fn persist(&self, data: &[u8], hint: ArtifactHint) -> Result<ArtifactRef, StoreError> {
        let hash = Sha256::digest(data);
        let id: String = hash[..8].iter().map(|byte| format!("{byte:02x}")).collect();
        let id = format!("sha256-{id}");
        let mut table = self
            .0
            .lock()
            .map_err(|e| StoreError::Persist(e.to_string()))?;
        table.insert(id.clone(), data.to_vec());
        Ok(ArtifactRef::new(id, data.len(), hint.kind, true))
    }

    async fn read(
        &self,
        id: &str,
        _range: Option<std::ops::Range<u64>>,
    ) -> Result<Vec<u8>, StoreError> {
        let table = self.0.lock().map_err(|e| StoreError::Read(e.to_string()))?;
        table
            .get(id)
            .cloned()
            .ok_or_else(|| StoreError::Read(format!("no asset {id}")))
    }
}

impl MediaResolver for Assets {
    // Called for every reference in every request, so it answers from
    // memory and never fetches.
    fn resolve(&self, media: &MediaRef) -> Option<MediaPayload> {
        let table = self.0.lock().ok()?;
        let bytes = table.get(&media.reference)?;
        let data = base64::engine::general_purpose::STANDARD.encode(bytes);
        Some(MediaPayload::inline(media.media_type.clone(), data))
    }
}
// ANCHOR_END: assets

// ANCHOR: tool
/// Draws a chart. The bytes go to the store; the result carries a
/// reference to them.
struct DrawChart;

#[async_trait]
impl Tool for DrawChart {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            "draw_chart",
            "Draw the sales chart.",
            json!({"type": "object"}),
        )
    }

    async fn execute(
        &self,
        ctx: &ToolCallContext,
        _control: &CallControl,
        store: Option<&dyn ArtifactStore>,
    ) -> ToolResultPayload {
        let png = draw_png();
        let failed = |error: String| {
            ToolResultPayload::new(
                ctx.call_block_id,
                ToolResultStatus::Failed,
                ToolOutput::new(json!({ "error": error })),
            )
        };
        let Some(store) = store else {
            return failed("no asset store configured".into());
        };
        let hint = ArtifactHint::new("draw_chart", ctx.call_block_id, ArtifactKind::Binary);
        match store.persist(&png, hint).await {
            Ok(artifact) => ToolResultPayload::new(
                ctx.call_block_id,
                ToolResultStatus::Succeeded,
                ToolOutput::new(json!("The chart is attached.")),
            )
            .with_media(vec![MediaRef::new("image/png", artifact.id)]),
            Err(error) => failed(error.to_string()),
        }
    }
}
// ANCHOR_END: tool

/// A one-pixel PNG, standing in for a chart.
fn draw_png() -> Vec<u8> {
    base64::engine::general_purpose::STANDARD
        .decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")
        .unwrap_or_default()
}

/// The parts of each message or item in a request body, one line each.
fn parts(body: &Value) -> String {
    fn shape(item: &Value) -> String {
        let label = item["role"]
            .as_str()
            .or(item["type"].as_str())
            .unwrap_or("?");
        let inner = item.get("content").or(item.get("output"));
        match inner {
            Some(Value::Array(parts)) => {
                let parts: Vec<String> = parts.iter().map(shape).collect();
                format!("{label}[{}]", parts.join(", "))
            }
            Some(Value::String(text)) if text.starts_with("[media") => {
                format!("{label}[{text}]")
            }
            Some(Value::String(_)) => format!("{label}[text]"),
            _ => match item["text"].as_str() {
                Some(text) if text.starts_with("[media") => format!("{label} {text}"),
                _ => label.to_owned(),
            },
        }
    }
    let items = body
        .get("messages")
        .or(body.get("input"))
        .and_then(Value::as_array);
    let lines: Vec<String> = items
        .into_iter()
        .flatten()
        .map(|item| format!("  {}", shape(item)))
        .collect();
    lines.join("\n")
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let assets = Arc::new(Assets::default());
    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(DrawChart)];

    // ANCHOR: executor
    let options = ToolExecutorOptions::default().with_artifact_store(assets.clone());
    let executor = Arc::new(ToolExecutor::new(tools, options)?);
    // ANCHOR_END: executor

    // A first turn, with a scripted model, in which the tool draws the chart.
    let scripted = Arc::new(ScriptedModel(|request: &_| {
        match tool_results(request).last() {
            None => call("draw_chart", json!({})),
            Some(_) => reply("Here is the chart."),
        }
    }));
    let context = Context::from_blocks(vec![ContextBlock::text(
        new_block_id(),
        Role::User,
        "Draw the sales chart.",
    )])?;
    let outcome = TurnRunner::new(scripted, executor.clone())
        .run(
            TurnId::new("chart"),
            context,
            TurnRunOptions::new(ModelRef::new("scripted")),
            CallControl::new(CancellationToken::new(), None),
        )
        .await;
    let mut context = outcome.context;

    // ANCHOR: user_media
    // A file the user uploads: store the bytes, put the reference in the
    // message.
    let pdf = b"%PDF-1.4 a quarterly report".to_vec();
    let hint = ArtifactHint::new("upload", new_block_id(), ArtifactKind::Binary);
    let stored = assets.persist(&pdf, hint).await?;
    let report = MediaRef::new("application/pdf", stored.id).with_name("report.pdf");
    context
        .edit()
        .append([ContextBlock::new(
            new_block_id(),
            BlockContent::Message {
                role: Role::User,
                parts: vec![
                    ContentPart::Text(TextPayload::new("Compare it with this report.")),
                    ContentPart::Media(report),
                ],
            },
            BlockMeta::default(),
        )])
        .commit()?;
    // ANCHOR_END: user_media

    // The next request, rendered by each gateway.
    let server = FakeProvider::start().await?;
    let base = server.base_url.clone();
    // ANCHOR: resolver
    let gateways: [(&str, Arc<dyn ModelGateway>); 3] = [
        (
            "Anthropic Messages",
            Arc::new(
                AnthropicMessagesGateway::new("test-key")
                    .with_base_url(&base)
                    .with_media_resolver(assets.clone()),
            ),
        ),
        (
            "Chat Completions",
            Arc::new(
                OpenAiChatCompletionsGateway::new("test-key")
                    .with_base_url(&base)
                    .with_media_resolver(assets.clone()),
            ),
        ),
        (
            "Responses",
            Arc::new(
                OpenAiResponsesGateway::new("test-key")
                    .with_base_url(&base)
                    .with_media_resolver(assets.clone()),
            ),
        ),
    ];
    // ANCHOR_END: resolver
    for (name, gateway) in gateways {
        let options = TurnRunOptions::new(ModelRef::new("model"))
            .with_generation(GenerationOptions::default().with_max_tokens(200));
        let _ = TurnRunner::new(gateway, executor.clone())
            .run(
                TurnId::new("compare"),
                context.clone(),
                options,
                CallControl::new(CancellationToken::new(), None),
            )
            .await;
        let sent = server.last().ok_or("no request reached the server")?;
        println!("{name}:\n{}", parts(&sent.body));
    }
    Ok(())
}
