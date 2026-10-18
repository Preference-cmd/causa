// The "Choose a protocol and provider" guide. Runs offline: a local server
// answers like the three provider APIs, and the program sends the same
// context through each gateway, built from a configuration value, and
// prints what each one sent.

// ANCHOR: imports
use causa::{
    kernel::{
        CallControl, CancellationToken, Context, ContextBlock, GenerationOptions, ModelGateway,
        ModelRef, Role, TurnId,
    },
    providers::{
        AnthropicMessagesGateway, AnthropicRenderOptions, ChatRenderOptions,
        OpenAiChatCompletionsGateway, OpenAiResponsesGateway, ResponsesRenderOptions,
    },
    runtime::{
        ToolExecutor, ToolExecutorOptions, TurnResult, TurnRunOptions, TurnRunner, new_block_id,
    },
};
use serde::Deserialize;
use std::sync::Arc;
// ANCHOR_END: imports

use causa_site_guides::fake_provider::FakeProvider;

// ANCHOR: config
/// One model endpoint, as the host's configuration file describes it.
/// The API key is not in it: read that from where you keep secrets.
#[derive(Deserialize)]
struct ProviderConfig {
    protocol: Protocol,
    /// A compatible server, with its version segment; omit for the
    /// protocol's own API.
    base_url: Option<String>,
    model: String,
    #[serde(default)]
    generation: GenerationOptions,
    #[serde(default)]
    anthropic: AnthropicRenderOptions,
    #[serde(default)]
    chat: ChatRenderOptions,
    #[serde(default)]
    responses: ResponsesRenderOptions,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Protocol {
    Anthropic,
    Chat,
    Responses,
}
// ANCHOR_END: config

// ANCHOR: gateway
fn gateway(config: &ProviderConfig, api_key: String) -> Arc<dyn ModelGateway> {
    match config.protocol {
        Protocol::Anthropic => {
            let mut gateway =
                AnthropicMessagesGateway::new(api_key).with_render_options(config.anthropic);
            if let Some(url) = &config.base_url {
                gateway = gateway.with_base_url(url);
            }
            Arc::new(gateway)
        }
        Protocol::Chat => {
            let mut gateway =
                OpenAiChatCompletionsGateway::new(api_key).with_render_options(config.chat);
            if let Some(url) = &config.base_url {
                gateway = gateway.with_base_url(url);
            }
            Arc::new(gateway)
        }
        Protocol::Responses => {
            let mut gateway =
                OpenAiResponsesGateway::new(api_key).with_render_options(config.responses);
            if let Some(url) = &config.base_url {
                gateway = gateway.with_base_url(url);
            }
            Arc::new(gateway)
        }
    }
}
// ANCHOR_END: gateway

// ANCHOR: files
// Three configuration files. `{base}` stands for the server's URL, such
// as `https://llm.example.com/v1`.
const CONFIGS: [&str; 3] = [
    r#"{ "protocol": "anthropic", "base_url": "{base}", "model": "model-a",
         "generation": { "max_tokens": 200 } }"#,
    r#"{ "protocol": "chat", "base_url": "{base}", "model": "model-b",
         "generation": { "max_tokens": 200 },
         "chat": { "max_tokens_field": "max_tokens" } }"#,
    r#"{ "protocol": "responses", "base_url": "{base}", "model": "model-c",
         "generation": { "max_tokens": 200 } }"#,
];
// ANCHOR_END: files

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let server = FakeProvider::start().await?;

    for text in CONFIGS {
        let config: ProviderConfig =
            serde_json::from_str(&text.replace("{base}", &server.base_url))?;

        // ANCHOR: run
        let runner = TurnRunner::new(
            gateway(&config, "test-key".into()),
            Arc::new(ToolExecutor::new(
                Vec::new(),
                ToolExecutorOptions::default(),
            )?),
        );
        let context = Context::from_blocks(vec![ContextBlock::text(
            new_block_id(),
            Role::User,
            "Say hello.",
        )])?;
        let options = TurnRunOptions::new(ModelRef::new(config.model.clone()))
            .with_generation(config.generation.clone());
        let outcome = runner
            .run(
                TurnId::new("hello"),
                context,
                options,
                CallControl::new(CancellationToken::new(), None),
            )
            .await;
        // ANCHOR_END: run

        let reply = match outcome.result {
            TurnResult::Completed { final_output } => final_output.response.text(),
            result => return Err(format!("turn did not complete: {result:?}").into()),
        };
        let sent = server.last().ok_or("no request reached the server")?;
        let limit = ["max_tokens", "max_completion_tokens", "max_output_tokens"]
            .into_iter()
            .find(|key| sent.body.get(*key).is_some())
            .unwrap_or("no limit");
        println!(
            "{} {} {limit}={} -> {reply}",
            sent.path, sent.auth_header, sent.body[limit]
        );
    }
    Ok(())
}
