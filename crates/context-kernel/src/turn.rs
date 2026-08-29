//! TurnContext / ContextFrame / WindowBudget / Compaction — Slice 1 single-turn kernel.
use crate::block::{
    BlockPayload, ContextBlock, InputPayload, TextPayload, ToolCallPayload, ToolResultPayload,
};
use crate::ids::{BlockId, BlockSequence, ContextVersion, FrameId, RoundId, TurnId, TurnSequence};
use crate::model::ModelOutput;
use crate::tool_data::ToolCallId;
use std::collections::HashSet;
use std::sync::Arc;

pub use crate::budget::{
    Compaction, CompactionError, CompactionInput, CompactionOutput, NoopCompaction,
    NoopTokenCounter, TokenCounter, WindowBudget,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnLifecycle {
    Open,
    Sealed,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum FrameScope {
    Turn {
        turn_id: TurnId,
        source_version: ContextVersion,
    },
}

#[derive(Debug, Clone)]
pub struct ModelContext {
    pub blocks: Vec<ContextBlock>,
}

#[derive(Debug, Clone)]
pub struct ContextFrame {
    pub frame_id: FrameId,
    pub scope: FrameScope,
    pub round_id: RoundId,
    pub model_context: ModelContext,
}

#[derive(Debug, thiserror::Error)]
pub enum ContextError {
    #[error("sealed turn")]
    SealedTurn,
    #[error("invalid sequence: {0}")]
    InvalidSequence(String),
    #[error("duplicate tool call id: {0:?}")]
    DuplicateToolCallId(crate::tool_data::ToolCallId),
    #[error("unpaired tool result: {0:?}")]
    UnpairedToolResult(crate::tool_data::ToolCallId),
    #[error("invalid model output: {0}")]
    InvalidModelOutput(String),
}

#[derive(Debug, thiserror::Error)]
pub enum FrameError {
    #[error("compaction failed: {0}")]
    CompactionFailed(String),
}

#[derive(Debug, Clone)]
pub struct AppliedModelOutput {
    pub block_ids: Vec<BlockId>,
    pub invocation_id: crate::ids::InvocationId,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OrderedBlocks(Vec<ContextBlock>);
impl OrderedBlocks {
    pub fn empty() -> Self {
        Self(Vec::new())
    }
    pub fn as_slice(&self) -> &[ContextBlock] {
        &self.0
    }
}

/// Current turn's ordered fact state. All mutation goes through the controlled
/// operations (`append_input` / `apply_model_output` / `append_tool_results` /
/// `seal`); there is no second `&mut` seam — fields are private by design.
pub struct TurnContext {
    turn_id: TurnId,
    blocks: OrderedBlocks,
    version: ContextVersion,
    lifecycle: TurnLifecycle,
    window_budget: WindowBudget,
    compaction: Option<Arc<dyn Compaction>>,
    token_counter: Option<Arc<dyn TokenCounter>>,
    next_seq: u64,
}

impl std::fmt::Debug for TurnContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TurnContext")
            .field("turn_id", &self.turn_id)
            .field("version", &self.version)
            .field("lifecycle", &self.lifecycle)
            .field("blocks_len", &self.blocks.0.len())
            .field("next_seq", &self.next_seq)
            .finish()
    }
}

fn input_to_payload(payload: InputPayload) -> BlockPayload {
    match payload {
        InputPayload::InstructionSystem(p) => BlockPayload::InstructionSystem(p),
        InputPayload::ContextInject(p) => BlockPayload::ContextInject(p),
        InputPayload::RequestUser(p) => BlockPayload::RequestUser(p),
    }
}

impl TurnContext {
    pub fn new(turn_id: TurnId) -> Self {
        Self {
            turn_id,
            blocks: OrderedBlocks::empty(),
            version: ContextVersion(0),
            lifecycle: TurnLifecycle::Open,
            window_budget: WindowBudget::default(),
            compaction: None,
            token_counter: None,
            next_seq: 0,
        }
    }
    pub fn with_window_budget(mut self, b: WindowBudget) -> Self {
        assert!(
            b.compaction_trigger < b.model_window_limit,
            "compaction_trigger must be less than model_window_limit"
        );
        self.window_budget = b;
        self
    }
    pub fn with_compaction(mut self, c: Arc<dyn Compaction>) -> Self {
        self.compaction = Some(c);
        self
    }
    pub fn with_token_counter(mut self, c: Arc<dyn TokenCounter>) -> Self {
        self.token_counter = Some(c);
        self
    }

    pub fn is_sealed(&self) -> bool {
        matches!(self.lifecycle, TurnLifecycle::Sealed)
    }
    pub fn blocks(&self) -> &[ContextBlock] {
        &self.blocks.0
    }
    pub fn version(&self) -> ContextVersion {
        self.version
    }
    pub fn turn_id(&self) -> TurnId {
        self.turn_id.clone()
    }
    pub fn lifecycle(&self) -> TurnLifecycle {
        self.lifecycle
    }
    pub fn snapshot_blocks(&self) -> Vec<ContextBlock> {
        self.blocks.0.clone()
    }

    pub fn estimate_tokens(&self) -> usize {
        if let Some(counter) = &self.token_counter {
            counter.estimate(&self.blocks.0)
        } else {
            self.blocks
                .0
                .iter()
                .map(|b| {
                    serde_json::to_string(&b.payload)
                        .map(|s| s.len() / 4)
                        .unwrap_or(0)
                })
                .sum()
        }
    }

    pub fn append_input(&mut self, payload: InputPayload) -> Result<BlockId, ContextError> {
        if self.is_sealed() {
            return Err(ContextError::SealedTurn);
        }
        let block_payload = input_to_payload(payload);
        let id = self.push_block(block_payload);
        self.version = ContextVersion(self.version.0 + 1);
        Ok(id)
    }

    pub fn apply_inputs(
        &mut self,
        payloads: Vec<InputPayload>,
    ) -> Result<Vec<BlockId>, ContextError> {
        let mut ids = Vec::with_capacity(payloads.len());
        for p in payloads {
            ids.push(self.append_input(p)?);
        }
        Ok(ids)
    }

    pub fn apply_model_output(
        &mut self,
        invocation: crate::ids::InvocationId,
        output: ModelOutput,
    ) -> Result<AppliedModelOutput, ContextError> {
        if self.is_sealed() {
            return Err(ContextError::SealedTurn);
        }
        // Structural validation only — any stop reason is recordable as
        // facts; interpreting terminal reasons (e.g. interrupting on
        // MaxTokens/Refusal) is driver policy above the kernel.
        match output.stop_reason {
            crate::model::ModelStopReason::EndTurn => {
                if !output.assistant.tool_calls.is_empty() {
                    return Err(ContextError::InvalidSequence(
                        "EndTurn must have empty tool_calls".into(),
                    ));
                }
            }
            crate::model::ModelStopReason::ToolUse => {
                if output.assistant.tool_calls.is_empty() {
                    return Err(ContextError::InvalidSequence(
                        "ToolUse must have non-empty tool_calls".into(),
                    ));
                }
            }
            crate::model::ModelStopReason::MaxTokens | crate::model::ModelStopReason::Refusal => {}
        }
        // Validate tool drafts
        for draft in &output.assistant.tool_calls {
            if draft.tool_name.trim().is_empty() {
                return Err(ContextError::InvalidSequence("tool_name empty".into()));
            }
            if !draft.arguments.is_object() {
                return Err(ContextError::InvalidSequence(
                    "arguments must be object".into(),
                ));
            }
        }
        // Check duplicate call ids that would be generated in this batch
        let mut seen_call_ids = HashSet::new();
        for (pos, draft) in output.assistant.tool_calls.iter().enumerate() {
            let cid =
                ToolCallId::generate(invocation.round_id, &draft.tool_name, &draft.arguments, pos);
            if !seen_call_ids.insert(cid.clone()) {
                return Err(ContextError::DuplicateToolCallId(cid));
            }
            // also check against existing blocks duplicate
            for b in &self.blocks.0 {
                if let BlockPayload::ToolCall(tc) = &b.payload {
                    if tc.call_id == cid {
                        return Err(ContextError::DuplicateToolCallId(cid));
                    }
                }
            }
        }

        let mut block_ids = Vec::new();
        // ResponseAssistant if text non-empty (trimmed)
        let text_trimmed = output.assistant.text.0.trim().to_string();
        if !text_trimmed.is_empty() {
            let id = self.push_block(BlockPayload::ResponseAssistant(TextPayload(
                output.assistant.text.0.clone(),
            )));
            block_ids.push(id);
        }
        // ToolCall blocks
        for (pos, draft) in output.assistant.tool_calls.into_iter().enumerate() {
            let call_id =
                ToolCallId::generate(invocation.round_id, &draft.tool_name, &draft.arguments, pos);
            let payload = ToolCallPayload {
                call_id: call_id.clone(),
                tool_name: draft.tool_name,
                arguments: draft.arguments,
                provider_call_id: draft.provider_call_id,
            };
            let id = self.push_block(BlockPayload::ToolCall(payload));
            block_ids.push(id);
        }
        self.version = ContextVersion(self.version.0 + 1);
        Ok(AppliedModelOutput {
            block_ids,
            invocation_id: invocation,
        })
    }

    pub fn append_tool_results(
        &mut self,
        results: &[ToolResultPayload],
    ) -> Result<Vec<BlockId>, ContextError> {
        if self.is_sealed() {
            return Err(ContextError::SealedTurn);
        }
        if results.is_empty() {
            return Ok(vec![]);
        }
        // gather existing tool call ids and paired set
        let mut call_ids_present: HashSet<ToolCallId> = HashSet::new();
        let mut paired: HashSet<ToolCallId> = HashSet::new();
        for b in &self.blocks.0 {
            match &b.payload {
                BlockPayload::ToolCall(tc) => {
                    call_ids_present.insert(tc.call_id.clone());
                }
                BlockPayload::ToolResult(tr) => {
                    paired.insert(tr.call_id.clone());
                }
                _ => {}
            }
        }
        // validate all results atomically
        let mut to_check_paired = HashSet::new();
        for r in results {
            if !call_ids_present.contains(&r.call_id) {
                return Err(ContextError::UnpairedToolResult(r.call_id.clone()));
            }
            if paired.contains(&r.call_id) {
                return Err(ContextError::InvalidSequence(format!(
                    "tool call already paired: {:?}",
                    r.call_id
                )));
            }
            if !to_check_paired.insert(r.call_id.clone()) {
                return Err(ContextError::InvalidSequence(format!(
                    "duplicate tool result in batch: {:?}",
                    r.call_id
                )));
            }
        }
        // all valid, append
        let mut ids = Vec::with_capacity(results.len());
        for r in results {
            let id = self.push_block(BlockPayload::ToolResult(ToolResultPayload {
                call_id: r.call_id.clone(),
                status: r.status.clone(),
                output: r.output.clone(),
            }));
            ids.push(id);
        }
        self.version = ContextVersion(self.version.0 + 1);
        Ok(ids)
    }

    pub fn frame_sync(&self, round_id: RoundId) -> Result<ContextFrame, FrameError> {
        let frame_id = crate::ids::FrameId::deterministic(&self.turn_id, self.version, round_id);
        Ok(ContextFrame {
            frame_id,
            scope: FrameScope::Turn {
                turn_id: self.turn_id.clone(),
                source_version: self.version,
            },
            round_id,
            model_context: ModelContext {
                blocks: self.blocks.0.clone(),
            },
        })
    }

    pub async fn frame(&self, round_id: RoundId) -> Result<ContextFrame, FrameError> {
        // estimate tokens
        let estimated = self.estimate_tokens();
        if self.window_budget.should_compact(estimated) {
            if let Some(comp) = &self.compaction {
                let input = CompactionInput {
                    blocks: self.blocks.0.clone(),
                    budget: self.window_budget,
                    estimated_tokens: estimated,
                };
                let out = comp
                    .compact(input)
                    .await
                    .map_err(|e| FrameError::CompactionFailed(e.to_string()))?;
                let frame_id =
                    crate::ids::FrameId::deterministic(&self.turn_id, self.version, round_id);
                let blocks = out.blocks;
                return Ok(ContextFrame {
                    frame_id,
                    scope: FrameScope::Turn {
                        turn_id: self.turn_id.clone(),
                        source_version: self.version,
                    },
                    round_id,
                    model_context: ModelContext { blocks },
                });
            }
        }
        self.frame_sync(round_id)
    }

    pub(crate) fn seal(&mut self) {
        self.lifecycle = TurnLifecycle::Sealed;
    }

    fn push_block(&mut self, payload: BlockPayload) -> BlockId {
        let seq = BlockSequence(self.next_seq);
        let id = BlockId {
            turn_id: self.turn_id.clone(),
            sequence: seq,
        };
        let block = ContextBlock {
            id: id.clone(),
            sequence: seq,
            meta: crate::block::BlockMeta::default(),
            payload,
        };
        self.blocks.0.push(block);
        self.next_seq += 1;
        id
    }

    pub fn from_validated_blocks(
        turn_id: TurnId,
        blocks: Vec<ContextBlock>,
        version: ContextVersion,
    ) -> Result<Self, ContextError> {
        // validate monotonic sequence and turn_id matching
        for (idx, b) in blocks.iter().enumerate() {
            if b.id.turn_id != turn_id {
                return Err(ContextError::InvalidSequence(format!(
                    "block turn_id mismatch at {}",
                    idx
                )));
            }
            if b.sequence.0 != idx as u64 {
                return Err(ContextError::InvalidSequence(format!(
                    "block sequence not monotonic at {} expected {} got {}",
                    idx, idx, b.sequence.0
                )));
            }
            if b.id.sequence != b.sequence {
                return Err(ContextError::InvalidSequence(format!(
                    "block id sequence mismatch at {}",
                    idx
                )));
            }
        }
        // validate tool pairing: no duplicate call ids, every result paired
        let mut call_set = HashSet::new();
        let mut seen_results = HashSet::new();
        for b in &blocks {
            match &b.payload {
                BlockPayload::ToolCall(tc) => {
                    if !call_set.insert(tc.call_id.clone()) {
                        return Err(ContextError::DuplicateToolCallId(tc.call_id.clone()));
                    }
                }
                BlockPayload::ToolResult(tr) => {
                    if !call_set.contains(&tr.call_id) {
                        return Err(ContextError::UnpairedToolResult(tr.call_id.clone()));
                    }
                    if !seen_results.insert(tr.call_id.clone()) {
                        return Err(ContextError::InvalidSequence(format!(
                            "duplicate tool result for {:?}",
                            tr.call_id
                        )));
                    }
                }
                _ => {}
            }
        }
        let next_seq = blocks.len() as u64;
        Ok(Self {
            turn_id,
            blocks: OrderedBlocks(blocks),
            version,
            lifecycle: TurnLifecycle::Open,
            window_budget: WindowBudget::default(),
            compaction: None,
            token_counter: None,
            next_seq,
        })
    }

    pub fn snapshot(&self) -> TurnSnapshot {
        TurnSnapshot {
            turn_id: self.turn_id.clone(),
            turn_sequence: TurnSequence(0),
            blocks: self.blocks.clone(),
            source_version: self.version,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TurnSnapshot {
    pub turn_id: TurnId,
    pub turn_sequence: TurnSequence,
    pub blocks: OrderedBlocks,
    pub source_version: ContextVersion,
}
