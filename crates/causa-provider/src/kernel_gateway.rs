//! HTTP gateways share preparation; protocol state machines stay transport-free.

mod config;
mod gateway;
mod stream;

pub use config::{
    AnthropicGatewayConfig, KernelGatewayConfig, OpenAiChatGatewayConfig,
    OpenAiResponsesGatewayConfig,
};
pub use gateway::KernelHttpGateway;

/// Anthropic Messages adapter for the kernel's model gateway port.
pub type AnthropicMessagesGateway = KernelHttpGateway<AnthropicGatewayConfig>;
/// OpenAI Chat Completions adapter for the kernel's model gateway port.
pub type OpenAiChatCompletionsGateway = KernelHttpGateway<OpenAiChatGatewayConfig>;
/// OpenAI Responses adapter for the kernel's model gateway port.
pub type OpenAiResponsesGateway = KernelHttpGateway<OpenAiResponsesGatewayConfig>;
