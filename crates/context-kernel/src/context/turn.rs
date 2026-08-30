//! TurnContext / ContextFrame / TurnSnapshot — Slice 1 single-turn fact machine.
use crate::context::block::{BlockContent, BlockMeta, ContextBlock, TextPayload, ToolCallPayload};
use crate::context::ids::{
    BlockId, BlockSequence, ContextVersion, FrameId, RoundId, TurnId, TurnSequence,
};
use crate::context::model::{ModelResponse, ModelStopReason};
use crate::context::tool_data::{ToolCallId, ToolResultPayload};
use std::collections::{HashMap, HashSet};

use crate::ports::budget::{CompactionInput, FramePolicy};

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
    #[error("foreign invocation: expected turn {expected:?}, got {actual:?}")]
    ForeignInvocation { expected: TurnId, actual: TurnId },
    #[error("invalid model output: {0}")]
    InvalidModelOutput(String),
    #[error("invalid sequence: {0}")]
    InvalidSequence(String),
    #[error("duplicate tool call id: {0:?}")]
    DuplicateToolCallId(crate::context::tool_data::ToolCallId),
    #[error("unpaired tool result: {0:?}")]
    UnpairedToolResult(crate::context::tool_data::ToolCallId),
}

#[derive(Debug, thiserror::Error)]
pub enum FrameError {
    #[error("compaction failed: {0}")]
    CompactionFailed(String),
}

/// Receipt of the model door: the committed fact block ids plus the prepared
/// tool calls in model draft order. The tool calls are the execution handoff —
/// call ids are generated exactly once, here, so the driver dispatches the
/// same payloads the kernel committed and never re-reads blocks.
#[derive(Debug, Clone)]
pub struct AppliedModelOutput {
    pub block_ids: Vec<BlockId>,
    pub tool_calls: Vec<ToolCallPayload>,
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

/// Current turn's ordered fact state. All mutation goes through the three
/// controlled doors (`append_input` / `append_model_output` /
/// `append_tool_results`) plus `seal`; there is no second `&mut` seam —
/// fields are private by design.
/// Frame-materialization policy (budget/compaction/counter) is NOT stored
/// here — facts stay facts; drivers pass a `FramePolicy` into `frame()`.
pub struct TurnContext {
    turn_id: TurnId,
    blocks: OrderedBlocks,
    version: ContextVersion,
    lifecycle: TurnLifecycle,
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

impl TurnContext {
    pub fn new(turn_id: TurnId) -> Self {
        Self {
            turn_id,
            blocks: OrderedBlocks::empty(),
            version: ContextVersion(0),
            lifecycle: TurnLifecycle::Open,
            next_seq: 0,
        }
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

    /// Host door: admit one text fact with its provenance label, recorded
    /// verbatim in the envelope. The kernel does not interpret the label;
    /// role assignment is the renderer's job.
    pub fn append_input(
        &mut self,
        text: TextPayload,
        source: impl Into<String>,
    ) -> Result<BlockId, ContextError> {
        self.ensure_open()?;
        let meta = BlockMeta {
            source: Some(source.into()),
            ..BlockMeta::default()
        };
        let mut ids = self.commit_blocks(vec![(BlockContent::Text(text), meta)]);
        Ok(ids.pop().expect("single-block commit yields one id"))
    }

    /// Model door: record the model's response as facts. Borrows the response
    /// so the caller keeps the envelope for its own traces; usage and
    /// reasoning are deliberately caller-retained and never enter the fact
    /// state. Tool call ids are generated exactly once and returned in the
    /// receipt as the execution handoff.
    pub fn append_model_output(
        &mut self,
        invocation: crate::context::ids::InvocationId,
        response: &ModelResponse,
        stop_reason: ModelStopReason,
    ) -> Result<AppliedModelOutput, ContextError> {
        self.ensure_open()?;
        if invocation.turn_id != self.turn_id {
            return Err(ContextError::ForeignInvocation {
                expected: self.turn_id.clone(),
                actual: invocation.turn_id.clone(),
            });
        }
        // Structural validation only — any stop reason is recordable as
        // facts; interpreting terminal reasons (e.g. interrupting on
        // MaxTokens/Refusal) is driver policy above the kernel.
        match stop_reason {
            ModelStopReason::EndTurn if !response.tool_calls.is_empty() => {
                return Err(ContextError::InvalidModelOutput(
                    "EndTurn must have empty tool_calls".into(),
                ));
            }
            ModelStopReason::ToolUse if response.tool_calls.is_empty() => {
                return Err(ContextError::InvalidModelOutput(
                    "ToolUse must have non-empty tool_calls".into(),
                ));
            }
            _ => {}
        }
        // Prepare everything before committing anything: text (when
        // non-empty), then one call payload per draft with its id generated
        // exactly once. Nothing touches the fact state until `commit_blocks`.
        let mut prepared: Vec<(BlockContent, BlockMeta)> = Vec::new();
        let mut tool_calls: Vec<ToolCallPayload> = Vec::new();
        if !response.text.0.trim().is_empty() {
            prepared.push((
                BlockContent::Text(TextPayload(response.text.0.clone())),
                BlockMeta::default(),
            ));
        }
        let mut seen: HashSet<ToolCallId> = HashSet::new();
        for (pos, draft) in response.tool_calls.iter().enumerate() {
            if draft.tool_name.trim().is_empty() {
                return Err(ContextError::InvalidModelOutput("tool_name empty".into()));
            }
            if !draft.arguments.is_object() {
                return Err(ContextError::InvalidModelOutput(
                    "arguments must be object".into(),
                ));
            }
            let call_id =
                ToolCallId::generate(invocation.round_id, &draft.tool_name, &draft.arguments, pos);
            if !seen.insert(call_id.clone()) {
                return Err(ContextError::DuplicateToolCallId(call_id));
            }
            for b in &self.blocks.0 {
                if let BlockContent::ToolCall(tc) = &b.content {
                    if tc.call_id == call_id {
                        return Err(ContextError::DuplicateToolCallId(call_id));
                    }
                }
            }
            tool_calls.push(ToolCallPayload {
                call_id: call_id.clone(),
                tool_name: draft.tool_name.clone(),
                arguments: draft.arguments.clone(),
            });
            // `provider_call_id` rides on the envelope `BlockMeta`; pairing
            // stays on the kernel-generated `call_id`.
            prepared.push((
                BlockContent::ToolCall(ToolCallPayload {
                    call_id: call_id.clone(),
                    tool_name: draft.tool_name.clone(),
                    arguments: draft.arguments.clone(),
                }),
                BlockMeta {
                    provider_call_id: draft.provider_call_id.clone(),
                    ..BlockMeta::default()
                },
            ));
        }
        let block_ids = self.commit_blocks(prepared);
        Ok(AppliedModelOutput {
            block_ids,
            tool_calls,
        })
    }

    /// Tool door: commit tool results. Canonical commit order is the paired
    /// call's block order (the model's original draft order), so two drivers
    /// submitting the same logical results in different completion orders
    /// produce identical snapshots. Batch completeness is runner policy; the
    /// kernel enforces pairing only.
    pub fn append_tool_results(
        &mut self,
        results: Vec<ToolResultPayload>,
    ) -> Result<Vec<BlockId>, ContextError> {
        self.ensure_open()?;
        if results.is_empty() {
            return Ok(vec![]);
        }
        // Pairing state from committed facts: each call's block sequence and
        // the set of already-paired calls.
        let mut call_seq: HashMap<ToolCallId, BlockSequence> = HashMap::new();
        let mut paired: HashSet<ToolCallId> = HashSet::new();
        for b in &self.blocks.0 {
            match &b.content {
                BlockContent::ToolCall(tc) => {
                    call_seq.insert(tc.call_id.clone(), b.sequence);
                }
                BlockContent::ToolResult(tr) => {
                    paired.insert(tr.call_id.clone());
                }
                _ => {}
            }
        }
        // Validate all results atomically, then commit in canonical order.
        let mut batch: HashSet<ToolCallId> = HashSet::new();
        let mut ordered: Vec<(BlockSequence, &ToolResultPayload)> = Vec::new();
        for r in &results {
            let seq = match call_seq.get(&r.call_id) {
                Some(seq) => *seq,
                None => return Err(ContextError::UnpairedToolResult(r.call_id.clone())),
            };
            if paired.contains(&r.call_id) {
                return Err(ContextError::InvalidSequence(format!(
                    "tool call already paired: {:?}",
                    r.call_id
                )));
            }
            if !batch.insert(r.call_id.clone()) {
                return Err(ContextError::InvalidSequence(format!(
                    "duplicate tool result in batch: {:?}",
                    r.call_id
                )));
            }
            ordered.push((seq, r));
        }
        ordered.sort_by_key(|(seq, _)| *seq);
        let prepared = ordered
            .into_iter()
            .map(|(_, r)| {
                (
                    BlockContent::ToolResult(ToolResultPayload {
                        call_id: r.call_id.clone(),
                        status: r.status.clone(),
                        output: r.output.clone(),
                    }),
                    BlockMeta::default(),
                )
            })
            .collect();
        Ok(self.commit_blocks(prepared))
    }

    pub fn frame_sync(&self, round_id: RoundId) -> Result<ContextFrame, FrameError> {
        let frame_id =
            crate::context::ids::FrameId::deterministic(&self.turn_id, self.version, round_id);
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

    /// Materialize the model context for `round_id`. Trigger *evaluation*
    /// stays canonical: the frame policy is a carrier of port instances —
    /// this method never references staged modules. Compaction output is
    /// frame-local and never written back into the fact state.
    pub async fn frame(
        &self,
        round_id: RoundId,
        policy: &FramePolicy,
    ) -> Result<ContextFrame, FrameError> {
        let estimated = policy.estimate(&self.blocks.0);
        if policy.window_budget.should_compact(estimated) {
            if let Some(comp) = &policy.compaction {
                let input = CompactionInput {
                    blocks: self.blocks.0.clone(),
                    budget: policy.window_budget,
                    estimated_tokens: estimated,
                };
                let out = comp
                    .compact(input)
                    .await
                    .map_err(|e| FrameError::CompactionFailed(e.to_string()))?;
                let frame_id = crate::context::ids::FrameId::deterministic(
                    &self.turn_id,
                    self.version,
                    round_id,
                );
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

    /// Terminal lifecycle transition, owned by the driver: seal when the turn
    /// is over. Sealed turns reject every append operation; the kernel
    /// guarantees no post-terminal mutation. (Conversation-level history
    /// eligibility is a Slice 2 driver stamp, not this marker.)
    pub fn seal(&mut self) {
        self.lifecycle = TurnLifecycle::Sealed;
    }

    fn ensure_open(&self) -> Result<(), ContextError> {
        if self.is_sealed() {
            Err(ContextError::SealedTurn)
        } else {
            Ok(())
        }
    }

    /// The single commit primitive: write prepared blocks, then bump the
    /// version exactly once iff the commit is non-empty. `ContextVersion`
    /// counts canonical fact commits — no-ops (empty output, empty results,
    /// failed validation, retries, compaction) leave it untouched.
    fn commit_blocks(&mut self, prepared: Vec<(BlockContent, BlockMeta)>) -> Vec<BlockId> {
        if prepared.is_empty() {
            return Vec::new();
        }
        let mut ids = Vec::with_capacity(prepared.len());
        for (content, meta) in prepared {
            ids.push(self.push_block(content, meta));
        }
        self.version = self.version.next();
        ids
    }

    fn push_block(&mut self, content: BlockContent, meta: BlockMeta) -> BlockId {
        let seq = BlockSequence(self.next_seq);
        let id = BlockId {
            turn_id: self.turn_id.clone(),
            sequence: seq,
        };
        let block = ContextBlock {
            id: id.clone(),
            sequence: seq,
            content,
            meta,
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
            match &b.content {
                BlockContent::ToolCall(tc) => {
                    if !call_set.insert(tc.call_id.clone()) {
                        return Err(ContextError::DuplicateToolCallId(tc.call_id.clone()));
                    }
                }
                BlockContent::ToolResult(tr) => {
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
