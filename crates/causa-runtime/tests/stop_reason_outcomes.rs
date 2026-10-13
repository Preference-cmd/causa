//! Every stop reason the kernel names is classified in one place: the runner
//! commits only EndTurn and ToolUse, and returns the complete uncommitted
//! output for the other four.
mod common;
use causa_kernel::*;
use causa_runtime::*;
use common::*;

/// One exhaustive match per stop reason. Adding a variant to the kernel enum
/// breaks this test until the new behaviour is stated here, which is the point.
fn outcome_of(reason: ModelStopReason) -> &'static str {
    match reason {
        ModelStopReason::EndTurn => "commit",
        ModelStopReason::ToolUse => "tool round",
        ModelStopReason::MaxTokens => "ModelMaxTokens",
        ModelStopReason::Refusal => "ModelRefusal",
        ModelStopReason::ContextWindowExceeded => "ModelContextWindowExceeded",
        ModelStopReason::Unrecognized => "ModelStopUnrecognized",
    }
}

fn cause(out: &TurnOutcome) -> &TurnInterruption {
    match &out.result {
        TurnResult::Interrupted { cause } => cause,
        _ => panic!("expected interruption"),
    }
}

#[tokio::test]
async fn every_stop_reason_is_committed_or_reported_whole() {
    for reason in [
        ModelStopReason::EndTurn,
        ModelStopReason::ToolUse,
        ModelStopReason::MaxTokens,
        ModelStopReason::Refusal,
        ModelStopReason::ContextWindowExceeded,
        ModelStopReason::Unrecognized,
    ] {
        let mut output = endturn_output("body the model produced");
        output.stop_reason = reason;
        output.raw_stop_reason = Some("wire-value".into());
        output.usage = Some(ModelUsage::new(7, 3));
        let gateway = RecordingGateway::scripted(vec![Ok(output)]);
        let out = runner_with(gateway, vec![])
            .run(TurnId::new("terminal"), Context::new(), options(), ctrl())
            .await;
        match outcome_of(reason) {
            "commit" => {
                assert!(matches!(out.result, TurnResult::Completed { .. }));
            }
            "tool round" => {
                // ToolUse with no declaration is a conversion failure carrying
                // the output, not a commit.
                let interruption = cause(&out);
                assert!(
                    matches!(interruption, TurnInterruption::ModelConversionFailed { .. }),
                    "{reason:?} produced {interruption:?}"
                );
                assert_eq!(out.context.blocks().len(), 0);
            }
            variant => {
                let interruption = cause(&out);
                match (variant, interruption) {
                    ("ModelMaxTokens", TurnInterruption::ModelMaxTokens { output, .. })
                    | ("ModelRefusal", TurnInterruption::ModelRefusal { output, .. })
                    | (
                        "ModelContextWindowExceeded",
                        TurnInterruption::ModelContextWindowExceeded { output, .. },
                    )
                    | (
                        "ModelStopUnrecognized",
                        TurnInterruption::ModelStopUnrecognized { output, .. },
                    ) => {
                        // The output, its usage and the wire value stay whole.
                        assert_eq!(output.response.text(), "body the model produced");
                        assert_eq!(
                            output.raw_stop_reason.as_deref(),
                            Some("wire-value"),
                            "{reason:?} must keep the wire stop value"
                        );
                        assert_eq!(output.usage.as_ref().map(|u| u.input_tokens), Some(7));
                    }
                    other => panic!("{reason:?} produced {other:?}"),
                }
                assert_eq!(
                    out.context.blocks().len(),
                    0,
                    "{reason:?} must not commit a block"
                );
            }
        }
    }
}

/// The provider's stop value is text of any length and content. It reaches
/// the host on the output, never through the interruption's message.
#[tokio::test]
async fn an_unrecognized_stop_value_stays_out_of_the_message() {
    let wire = format!("{}\nsecond line\u{1b}[31m", "x".repeat(4096));
    let mut output = endturn_output("body");
    output.stop_reason = ModelStopReason::Unrecognized;
    output.raw_stop_reason = Some(wire.clone());
    let gateway = RecordingGateway::scripted(vec![Ok(output)]);
    let out = runner_with(gateway, vec![])
        .run(TurnId::new("terminal"), Context::new(), options(), ctrl())
        .await;
    let interruption = cause(&out);
    let message = interruption.to_string();
    assert_eq!(
        message,
        "the provider stopped for a reason this version does not recognize"
    );
    assert!(!message.chars().any(char::is_control));
    let TurnInterruption::ModelStopUnrecognized { output, .. } = interruption else {
        panic!("produced {interruption:?}");
    };
    assert_eq!(output.raw_stop_reason.as_deref(), Some(wire.as_str()));
}

/// A normal-end stop value with tool calls present is not reclassified: the
/// runner reports the conversion failure and hands the output back whole.
#[tokio::test]
async fn end_turn_with_tool_calls_is_a_conversion_failure_carrying_the_output() {
    let mut output = tooluse_output("text", "echo", serde_json::json!({}));
    output.stop_reason = ModelStopReason::EndTurn;
    output.raw_stop_reason = Some("end_turn".into());
    let gateway = RecordingGateway::scripted(vec![Ok(output)]);
    let out = runner_with(gateway, vec![])
        .run(TurnId::new("mismatch"), Context::new(), options(), ctrl())
        .await;
    let TurnInterruption::ModelConversionFailed { error, output, .. } = cause(&out) else {
        panic!("typed conversion failure expected, got {:?}", out.result)
    };
    assert!(matches!(
        error,
        ModelBlockError::EndTurnWithToolCalls { count: 1, .. }
    ));
    assert_eq!(output.response.tool_calls().count(), 1);
    assert_eq!(output.raw_stop_reason.as_deref(), Some("end_turn"));
    assert_eq!(out.context.blocks().len(), 0);
}
