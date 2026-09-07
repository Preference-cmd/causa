//! Decision 6/7/8 behavior tests (Slice 13 Phase G): the recorded result
//! and the continuation action are separate — actions resolve through
//! host configuration and explicit per-call decisions by the ACTUAL
//! executed name, never a tool declaration, the draft's name, or a value
//! read out of a result body. Wire and resume-side cases live in
//! `serialization.rs` / `recoverable_interruption.rs`.

mod common;

use async_trait::async_trait;
use causa_kernel::{
    BlockContent, CallControl, Tool, ToolCallContext, ToolCallId, ToolCallPayload, ToolDefinition,
    ToolOutput, ToolResultPayload, ToolResultStatus, Truncation,
};
use causa_runtime::{
    BatchDecision, HookCtx, HookOutcome, ToolExecutor, ToolOutputLimits, ToolUseHook,
    TurnInteraction, TurnInterruption, TurnOutcome, TurnPolicy, TurnResult, TurnRunOptions,
    TurnRunner, UnknownOutcomeConfig, UnknownOutcomePolicy,
};
use common::{
    RecordingGateway, ctrl, ctx, draft, endturn_output, options_with_limits, runner_with,
    tooluse_calls_output,
};
use serde_json::json;
use std::sync::Arc;

// ---- fixtures ------------------------------------------------------------------

/// Returns `UnknownOutcome` tagged with its own name so facts can be
/// matched back to the executing tool without relying on call-id encoding.
macro_rules! unknown_tool {
    ($name:ident, $tool:literal) => {
        struct $name;
        #[async_trait]
        impl Tool for $name {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: $tool.into(),
                    description: $tool.into(),
                    parameters: json!({"type": "object"}),
                }
            }
            async fn execute(&self, ctx: &ToolCallContext, _c: &CallControl) -> ToolResultPayload {
                ToolResultPayload {
                    call_id: ctx.call_id.clone(),
                    status: ToolResultStatus::UnknownOutcome,
                    output: ToolOutput::new(json!({"tool": $tool})),
                    media: Vec::new(),
                }
            }
        }
    };
}

unknown_tool!(StopperTool, "stopper");
unknown_tool!(ContinuerTool, "continuer");
unknown_tool!(RealTool, "real");

/// Emits a 600-char output (~150 estimated tokens at the chars/4
/// fallback) so a 10-token limit truncates but a 10,000-token one does not.
struct SizedTool(&'static str);
#[async_trait]
impl Tool for SizedTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.0.into(),
            description: self.0.into(),
            parameters: json!({"type": "object"}),
        }
    }
    async fn execute(&self, ctx: &ToolCallContext, _c: &CallControl) -> ToolResultPayload {
        ToolResultPayload {
            call_id: ctx.call_id.clone(),
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(json!("x".repeat(600))),
            media: Vec::new(),
        }
    }
}

fn result_blocks(out: &TurnOutcome) -> Vec<&ToolResultPayload> {
    out.context
        .blocks()
        .iter()
        .filter_map(|b| match &b.content {
            BlockContent::ToolResult(r) => Some(r),
            _ => None,
        })
        .collect()
}

fn options_with_unknown(config: UnknownOutcomeConfig) -> TurnRunOptions {
    TurnRunOptions {
        policy: TurnPolicy {
            unknown_outcome: config,
            ..Default::default()
        },
        ..options_with_limits(5, 10)
    }
}

fn expect_unknown_interruption(out: &TurnOutcome) -> ToolCallId {
    match &out.result {
        TurnResult::Interrupted {
            cause: TurnInterruption::UnsafeUnknownOutcome { call_id },
        } => call_id.clone(),
        other => panic!("expected UnsafeUnknownOutcome, got {other:?}"),
    }
}

// ---- one batch carrying both actions -------------------------------------------

/// A mixed batch commits every result verbatim in draft order; a single
/// Stop among UnknownOutcome results interrupts afterwards and never
/// cancels or drops the Continue sibling.
#[tokio::test]
async fn mixed_batch_commits_both_then_interrupts_on_the_stop_one() {
    let c = ctx("t1");
    let gw = RecordingGateway::scripted(vec![
        Ok(tooluse_calls_output(
            "two unknown calls",
            vec![draft("stopper", json!({})), draft("continuer", json!({}))],
        )),
        Ok(endturn_output("unreachable: the turn stopped first")),
    ]);
    let runner = runner_with(gw, vec![Arc::new(StopperTool), Arc::new(ContinuerTool)]);
    let mut config = UnknownOutcomeConfig::default();
    config
        .overrides
        .insert("continuer".into(), UnknownOutcomePolicy::Continue);
    let out = runner.run(c, options_with_unknown(config), ctrl()).await;
    let call_id = expect_unknown_interruption(&out);
    // The interrupted call is the Stop-configured one, identified through
    // the trace's per-call executed names.
    let batch = out.trace.rounds[0].tool_batch.as_ref().unwrap();
    let stopper = batch
        .calls
        .iter()
        .find(|t| t.tool_name == "stopper")
        .expect("stopper traced");
    assert_eq!(&call_id, &stopper.call_id);
    // Both siblings committed verbatim, in draft order, still UnknownOutcome.
    let results = result_blocks(&out);
    assert_eq!(results.len(), 2);
    assert!(
        results
            .iter()
            .all(|r| r.status == ToolResultStatus::UnknownOutcome)
    );
    assert_eq!(results[0].output.content, json!({"tool": "stopper"}));
    assert_eq!(results[1].output.content, json!({"tool": "continuer"}));
}

// ---- the executed name decides, not the draft's --------------------------------

/// A hook that retargets every call to `real` — the model drafted
/// `alias`, but `real` is what executes.
struct RenameToReal;
#[async_trait]
impl ToolUseHook for RenameToReal {
    async fn apply(&self, mut calls: Vec<ToolCallPayload>, _ctx: &HookCtx<'_>) -> HookOutcome {
        for payload in &mut calls {
            payload.tool_name = "real".into();
        }
        HookOutcome::passthrough(calls)
    }
}

fn renamed_runner(gateway: Arc<RecordingGateway>, tools: Vec<Arc<dyn Tool>>) -> TurnRunner {
    TurnRunner::with_hook(
        gateway,
        Arc::new(ToolExecutor::from_vec(tools)),
        Arc::new(RenameToReal),
    )
}

/// The action resolves by the executed name: the draft's `alias` override
/// (Continue) must not save the turn when the executed `real` falls back
/// to Stop; an explicit `real` override does.
#[tokio::test]
async fn hook_rewritten_tool_name_decides_the_action() {
    let scripted = || {
        RecordingGateway::scripted(vec![
            Ok(tooluse_calls_output(
                "drafted alias",
                vec![draft("alias", json!({}))],
            )),
            Ok(endturn_output("done")),
        ])
    };

    // Draft name says Continue, executed name says Stop → the turn stops.
    let c = ctx("t1");
    let mut config = UnknownOutcomeConfig::default();
    config
        .overrides
        .insert("alias".into(), UnknownOutcomePolicy::Continue);
    let out = renamed_runner(scripted(), vec![Arc::new(RealTool)])
        .run(c, options_with_unknown(config), ctrl())
        .await;
    let call_id = expect_unknown_interruption(&out);
    let batch = out.trace.rounds[0].tool_batch.as_ref().unwrap();
    // Observation keeps the draft identity; the ACTION above already
    // proved the executed name decided (draft override said Continue).
    assert_eq!(batch.calls[0].tool_name, "alias");
    assert_eq!(&call_id, &batch.calls[0].call_id);

    // The executed name's own override applies: Continue → next round,
    // the recorded result still reads UnknownOutcome.
    let c = ctx("t2");
    let mut config = UnknownOutcomeConfig::default();
    config
        .overrides
        .insert("alias".into(), UnknownOutcomePolicy::Continue);
    config
        .overrides
        .insert("real".into(), UnknownOutcomePolicy::Continue);
    let out = renamed_runner(scripted(), vec![Arc::new(RealTool)])
        .run(c, options_with_unknown(config), ctrl())
        .await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    let results = result_blocks(&out);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].status, ToolResultStatus::UnknownOutcome);
    assert_eq!(results[0].output.content, json!({"tool": "real"}));
}

// ---- per-call decisions on precomputed results ---------------------------------

/// Rejects both same-name calls with precomputed UnknownOutcome results
/// and pins opposite per-call actions: `a` continues, `b` stops.
struct SplitUnknownHook;
#[async_trait]
impl ToolUseHook for SplitUnknownHook {
    async fn apply(&self, calls: Vec<ToolCallPayload>, _ctx: &HookCtx<'_>) -> HookOutcome {
        let mut outcome = HookOutcome::passthrough(Vec::new());
        for payload in calls {
            let which = if payload.arguments == json!({"which": "a"}) {
                "a"
            } else {
                "b"
            };
            outcome.rejected.push(ToolResultPayload {
                call_id: payload.call_id,
                status: ToolResultStatus::UnknownOutcome,
                output: ToolOutput::new(json!({"which": which})),
                media: Vec::new(),
            });
        }
        let id_of = |outcome: &HookOutcome, which: &str| {
            outcome
                .rejected
                .iter()
                .find(|r| r.output.content == json!({ "which": which }))
                .expect("tagged result")
                .call_id
                .clone()
        };
        let (a, b) = (id_of(&outcome, "a"), id_of(&outcome, "b"));
        outcome
            .with_unknown_decision(a, UnknownOutcomePolicy::Continue)
            .with_unknown_decision(b, UnknownOutcomePolicy::Stop)
    }
}

/// Two precomputed UnknownOutcome results from the SAME tool name keep
/// opposite explicit actions: both commit, the Stop one interrupts, the
/// Continue one is not cancelled. The per-call decisions override the
/// per-name default (which is Continue here, so omitted entries would
/// have completed the turn).
#[tokio::test]
async fn same_name_precomputed_unknowns_keep_per_call_actions() {
    let c = ctx("t1");
    let gw = RecordingGateway::scripted(vec![
        Ok(tooluse_calls_output(
            "two unk calls",
            vec![
                draft("unk", json!({"which": "a"})),
                draft("unk", json!({"which": "b"})),
            ],
        )),
        Ok(endturn_output("unreachable: b stopped the turn")),
    ]);
    let config = UnknownOutcomeConfig {
        default: UnknownOutcomePolicy::Continue,
        overrides: Default::default(),
    };
    let runner = TurnRunner::with_hook(
        gw,
        Arc::new(ToolExecutor::from_vec(vec![])),
        Arc::new(SplitUnknownHook),
    );
    let out = runner.run(c, options_with_unknown(config), ctrl()).await;
    let call_id = expect_unknown_interruption(&out);
    let results = result_blocks(&out);
    assert_eq!(results.len(), 2);
    let b = results
        .iter()
        .find(|r| r.output.content == json!({"which": "b"}))
        .expect("b result committed");
    assert_eq!(&call_id, &b.call_id);
    assert!(
        results
            .iter()
            .all(|r| r.status == ToolResultStatus::UnknownOutcome)
    );
}

/// A `BatchDecision::Reject` whose decision set omits every entry
/// resolves through the turn's unknown-outcome configuration; the
/// precomputed result commits verbatim either way and is never rewritten
/// into a success.
struct RejectUnknownResults;
#[async_trait]
impl TurnInteraction for RejectUnknownResults {
    async fn decide_batch(&self, calls: &[ToolCallPayload]) -> BatchDecision {
        BatchDecision::Reject {
            results: calls
                .iter()
                .map(|c| ToolResultPayload {
                    call_id: c.call_id.clone(),
                    status: ToolResultStatus::UnknownOutcome,
                    output: ToolOutput::new(json!({"precomputed": c.tool_name})),
                    media: Vec::new(),
                })
                .collect(),
            unknown_decisions: Vec::new(),
        }
    }
}

#[tokio::test]
async fn reject_entries_may_be_omitted_and_resolve_through_the_config() {
    let scripted = || {
        RecordingGateway::scripted(vec![
            Ok(tooluse_calls_output(
                "call unk",
                vec![draft("unk", json!({}))],
            )),
            Ok(endturn_output("done")),
        ])
    };
    let mut rejecting = options_with_limits(5, 10);
    rejecting.interaction = Arc::new(RejectUnknownResults);

    // Default Stop → the turn interrupts after committing the result.
    let c = ctx("t1");
    let out = runner_with(scripted(), vec![])
        .run(c, rejecting.clone(), ctrl())
        .await;
    expect_unknown_interruption(&out);
    let results = result_blocks(&out);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].status, ToolResultStatus::UnknownOutcome);
    assert_eq!(results[0].output.content, json!({"precomputed": "unk"}));

    // Continue by name → the turn proceeds; the recorded result is
    // unchanged.
    let c = ctx("t2");
    let mut config = UnknownOutcomeConfig::default();
    config
        .overrides
        .insert("unk".into(), UnknownOutcomePolicy::Continue);
    let mut options = rejecting.clone();
    options.policy.unknown_outcome = config;
    let out = runner_with(scripted(), vec![])
        .run(c, options, ctrl())
        .await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    assert_eq!(out.trace.rounds.len(), 2);
    let results = result_blocks(&out);
    assert_eq!(results[0].status, ToolResultStatus::UnknownOutcome);
}

// ---- output retention by executed name (Decision 8) ----------------------------

/// The per-name override is the chosen limit — it wins over the fallback
/// even when LARGER — resolved by the executed name before dispatch.
#[tokio::test]
async fn output_limit_override_by_executed_name_wins_over_the_fallback() {
    let c = ctx("t1");
    let gw = RecordingGateway::scripted(vec![
        Ok(tooluse_calls_output(
            "two sized calls",
            vec![draft("wide", json!({})), draft("narrow", json!({}))],
        )),
        Ok(endturn_output("done")),
    ]);
    let runner = runner_with(
        gw,
        vec![Arc::new(SizedTool("wide")), Arc::new(SizedTool("narrow"))],
    );
    let mut cfg = options_with_limits(5, 10);
    cfg.execution.tool_output_limits = ToolOutputLimits { max_tokens: 10 };
    cfg.execution
        .tool_output_limits_overrides
        .insert("wide".into(), ToolOutputLimits { max_tokens: 10_000 });
    let out = runner.run(c, cfg, ctrl()).await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    let results = result_blocks(&out);
    assert_eq!(results.len(), 2);
    for r in results {
        if r.output.content == json!("x".repeat(600)) {
            assert_eq!(
                r.output.truncation,
                Truncation::None,
                "wide keeps everything"
            );
        } else {
            assert_eq!(r.output.truncation, Truncation::Middle, "narrow truncates");
        }
    }
}
