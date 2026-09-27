//! Optional result-retention policy for tool output, notes, and media refs.

use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use causa_kernel::{
    ArtifactHint, ArtifactKind, ArtifactStore, ProcessorContext, ProcessorError, ToolBatch,
    ToolBatchProcessor, Truncation,
};

use crate::budget::TokenCounter;

/// Optional post-processor that truncates tool output to an estimated-token
/// budget while retaining notes and media references.
///
/// No instance is installed by default. If notes and media alone exceed the
/// budget (including the truncation notice), processing returns an error and
/// leaves the current entry unchanged. Earlier entries may already have been
/// processed and remain available in the batch. A `usize::MAX` budget means
/// unlimited and skips estimation for that result.
pub struct ToolOutputBudgetProcessor {
    max_tokens: usize,
    tool_limits: HashMap<String, usize>,
    token_counter: Option<Arc<dyn TokenCounter>>,
    artifact_store: Option<Arc<dyn ArtifactStore>>,
}

impl ToolOutputBudgetProcessor {
    /// Create a budget processor with the built-in protocol-visible text
    /// UTF-8 bytes/4 estimate, rounded up. This is opt-in and affects only
    /// tool results.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            tool_limits: HashMap::new(),
            token_counter: None,
            artifact_store: None,
        }
    }

    /// Use the host's token estimator instead of the built-in text estimate.
    pub fn with_token_counter(mut self, counter: Arc<dyn TokenCounter>) -> Self {
        self.token_counter = Some(counter);
        self
    }

    /// Spill original output through the configured artifact port before
    /// truncation. A store failure returns an error before changing the
    /// current result; earlier results may already have been processed.
    pub fn with_artifact_store(mut self, store: Arc<dyn ArtifactStore>) -> Self {
        self.artifact_store = Some(store);
        self
    }

    /// Set the budget for one effective tool name. This overrides the
    /// processor-wide limit for results produced by that tool.
    pub fn for_tool(mut self, tool_name: impl Into<String>, max_tokens: usize) -> Self {
        self.tool_limits.insert(tool_name.into(), max_tokens);
        self
    }

    fn estimate_text(&self, text: &str) -> usize {
        self.token_counter.as_ref().map_or_else(
            || text.len().div_ceil(4),
            |counter| counter.estimate_value(&serde_json::Value::String(text.to_owned())),
        )
    }

    fn estimate_result(
        &self,
        text: &str,
        notes: &[causa_kernel::TextPayload],
        media: &[causa_kernel::MediaRef],
    ) -> Result<usize, ProcessorError> {
        let mut visible = text.to_owned();
        if !notes.is_empty() {
            visible.push_str("\n\nNotes:");
        }
        for note in notes {
            visible.push_str("\n- ");
            visible.push_str(&note.0);
        }
        let text_tokens = self.estimate_text(&visible);
        let mut media_tokens = 0usize;
        if !media.is_empty() {
            let counter = self.token_counter.as_ref().ok_or_else(|| {
                ProcessorError::Failed(
                    "tool result includes media but no media token estimator is configured".into(),
                )
            })?;
            for item in media {
                let estimate = counter.estimate_media(item).ok_or_else(|| {
                    ProcessorError::Failed(format!(
                        "no token estimate is available for media type {:?}",
                        item.media_type
                    ))
                })?;
                media_tokens = media_tokens.saturating_add(estimate);
            }
        }
        Ok(text_tokens.saturating_add(media_tokens))
    }
}

impl std::fmt::Debug for ToolOutputBudgetProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolOutputBudgetProcessor")
            .field("max_tokens", &self.max_tokens)
            .field("tool_limits", &self.tool_limits)
            .field("token_counter", &self.token_counter.is_some())
            .field("artifact_store", &self.artifact_store.is_some())
            .finish()
    }
}

#[async_trait]
impl ToolBatchProcessor for ToolOutputBudgetProcessor {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        _ctx: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        for entry in batch.results_mut() {
            let Some((_, result)) = entry.result() else {
                continue;
            };
            let original_content = result.output.content.clone();
            let notes = result.notes.clone();
            let media = result.media.clone();
            let call_id = entry.call().call_block_id;
            let tool_name = entry.call().input.tool_name.clone();
            let full_output = serde_json::to_vec(&original_content)
                .map_err(|error| ProcessorError::Failed(error.to_string()))?;
            let original_text = match &original_content {
                serde_json::Value::String(text) => text.clone(),
                _ => String::from_utf8_lossy(&full_output).into_owned(),
            };
            let budget = self
                .tool_limits
                .get(&tool_name)
                .copied()
                .unwrap_or(self.max_tokens);
            if budget == usize::MAX {
                continue;
            }
            // Estimate the protocol-visible result body: output text followed
            // by its ordered note sections and any separately rendered media.
            // This is result-local and makes no claim about total request cost.
            let output_tokens = self.estimate_text(&original_text);
            if self.estimate_result(&original_text, &notes, &media)? <= budget {
                continue;
            }

            let minimum_tokens = self.estimate_result(TRUNCATION_NOTICE, &notes, &media)?;
            if minimum_tokens > budget {
                return Err(ProcessorError::Failed(format!(
                    "tool result notes and media require at least {minimum_tokens} estimated tokens, over budget {budget}"
                )));
            }

            let chars = original_text.chars().collect::<Vec<_>>();
            let mut low = 0usize;
            let mut high = chars.len();
            while low < high {
                let mid = low + (high - low).div_ceil(2);
                let candidate = middle(&chars, mid);
                if self.estimate_result(&candidate, &notes, &media)? <= budget {
                    low = mid;
                } else {
                    high = mid - 1;
                }
            }
            let mut retained = middle(&chars, low);
            // Tokenizers need not be monotonic. Verify the chosen candidate
            // and step down until it fits, or fail at the irreducible floor.
            while self.estimate_result(&retained, &notes, &media)? > budget {
                if low == 0 {
                    return Err(ProcessorError::Failed(format!(
                        "tool result notes and media require more than budget {budget}"
                    )));
                }
                low -= 1;
                retained = middle(&chars, low);
            }

            let artifact = if let Some(store) = &self.artifact_store {
                Some(
                    store
                        .persist(
                            &full_output,
                            ArtifactHint {
                                tool_name,
                                call_block_id: call_id,
                                kind: ArtifactKind::FullOutput,
                            },
                        )
                        .await
                        .map_err(|error| ProcessorError::Failed(error.to_string()))?,
                )
            } else {
                None
            };
            let output = entry.output_mut().expect("result is present");
            output.content = serde_json::Value::String(retained);
            output.truncation = Truncation::Middle;
            // Old references point at old content and cannot describe this
            // truncation. Keep only a newly stored artifact for this exact
            // original content.
            output.artifact = artifact;
            let meta = output.meta.get_or_insert(causa_kernel::ToolOutputMeta {
                duration_ms: None,
                original_tokens: None,
                extra: None,
            });
            meta.original_tokens = Some(output_tokens);
        }
        Ok(())
    }
}

const TRUNCATION_NOTICE: &str = "… output truncated …";

fn middle(chars: &[char], keep: usize) -> String {
    if keep >= chars.len() {
        return chars.iter().collect();
    }
    let head = (keep * 3) / 5;
    let tail = keep - head;
    let mut output = String::with_capacity(keep + TRUNCATION_NOTICE.len());
    output.extend(chars[..head].iter());
    output.push_str(TRUNCATION_NOTICE);
    output.extend(chars[chars.len().saturating_sub(tail)..].iter());
    output
}
