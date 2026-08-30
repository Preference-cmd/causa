//! Second-driver existence proof (Slice 1.5 gates 8 and 11): a minimal
//! single-shot agent loop — one model invocation, no tools — assembled ONLY
//! from the public root facade. The kernel exposes facts and ports; this
//! file must compile and run without the staged reference driver
//! (`TurnRunner`) and without any private module path.

use reimagine_context_kernel::{
    AttemptControl, AttemptNumber, CancellationToken, ContextError, ContextVersion, FramePolicy,
    GenerationOptions, InvocationId, ModelGateway, ModelInvokeError, ModelOutput, ModelRef,
    ModelRequest, ModelResponse, ModelStopReason, ModelUsage, ReasoningPayload, RoundId,
    RunControl, TextPayload, ToolSurface, TurnContext, TurnId, TurnSnapshot,
};

/// A gateway that returns one canned output and asserts the invocation
/// identity a well-formed driver must present: round 0, first attempt.
struct OneShotGateway {
    output: ModelOutput,
}

#[async_trait::async_trait]
impl ModelGateway for OneShotGateway {
    async fn invoke(
        &self,
        request: &ModelRequest,
        _control: &AttemptControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        assert_eq!(request.invocation_id.round_id, RoundId(0));
        assert_eq!(request.attempt, AttemptNumber(1));
        // the frame projection presents the recorded facts in order
        assert_eq!(request.frame.model_context.blocks.len(), 1);
        Ok(self.output.clone())
    }
}

#[tokio::test]
async fn external_single_shot_driver_assembles_from_root_facade() {
    let mut context = TurnContext::new(TurnId::new("ext-1"));
    context
        .append_input(TextPayload::new("hi"), "user")
        .unwrap();

    let gateway = OneShotGateway {
        output: ModelOutput {
            response: ModelResponse {
                text: TextPayload::new("hello"),
                tool_calls: vec![],
            },
            usage: Some(ModelUsage {
                input_tokens: 10,
                output_tokens: 2,
                cache_read_tokens: Some(4),
                cache_write_tokens: None,
                reasoning_tokens: Some(1),
            }),
            stop_reason: ModelStopReason::EndTurn,
            reasoning: Some(ReasoningPayload {
                text: "thinking".into(),
                signature: Some("sig".into()),
            }),
        },
    };

    // The external loop: materialize a frame through the canonical policy
    // port, invoke the model through the gateway port, apply the output as
    // facts, and seal the turn — no reference driver involved.
    let frame = context
        .frame(RoundId(0), &FramePolicy::default())
        .await
        .unwrap();
    let invocation = InvocationId {
        turn_id: context.turn_id(),
        round_id: RoundId(0),
    };
    let request = ModelRequest {
        invocation_id: invocation.clone(),
        attempt: AttemptNumber(1),
        frame,
        model: ModelRef::new("external-model"),
        tool_surface: ToolSurface::empty(),
        generation: GenerationOptions::default(),
    };
    let ctrl = RunControl::new(CancellationToken::new(), None);
    let output = gateway
        .invoke(&request, &ctrl.for_attempt(None))
        .await
        .unwrap();
    let applied = context
        .append_model_output(invocation, &output.response, output.stop_reason)
        .unwrap();
    context.seal();

    assert!(context.is_sealed());
    assert_eq!(applied.block_ids.len(), 1); // response text only
    assert_eq!(context.version(), ContextVersion(2));
    // facts round-trip losslessly through a snapshot
    let json = serde_json::to_string(&context.snapshot()).unwrap();
    let snapshot: TurnSnapshot = serde_json::from_str(&json).unwrap();
    assert_eq!(snapshot.blocks.as_slice().len(), 2);
    // sealed turn rejects further mutation
    assert!(matches!(
        context.append_input(TextPayload::new("more"), "user"),
        Err(ContextError::SealedTurn)
    ));
}
