//! Demand-driven HTTP streaming, with no background read task.

use crate::gateway_transport::{
    check_control, controlled, map_status_error, map_transport_error, read_error_response,
    send_response,
};
use causa_kernel::{CallControl, ModelInvokeError, ModelInvokeErrorKind, ModelStream, StreamDelta};
use causa_protocol::translation::streaming::StreamDecoder;
use futures_util::StreamExt;

/// Limit each local byte progression and periodically yield even when the
/// response or decoder remains ready. The budget persists across deltas.
const DECODE_WINDOW: usize = 8 * 1024;
const PROGRESS_BUDGET: usize = 32;

pub(super) async fn establish_stream(
    request: reqwest::RequestBuilder,
    control: &CallControl,
    provider: &str,
) -> Result<reqwest::Response, ModelInvokeError> {
    let response = send_response(request, control).await?;
    if !response.status().is_success() {
        let headers = response.headers().clone();
        let status = read_error_response(response, control).await?;
        // Error bodies may echo prompts, tool arguments or credentials. HTTP
        // classification reads the status and the retry headers only; keep
        // diagnostics generic.
        return Err(map_status_error(
            status,
            &headers,
            "response body omitted",
            provider,
        ));
    }
    let is_sse = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("text/event-stream"));
    if !is_sse {
        return Err(ModelInvokeError::new(
            ModelInvokeErrorKind::Permanent,
            "successful streaming response requires Content-Type text/event-stream",
        ));
    }
    check_control(control)?;
    Ok(response)
}

struct State {
    response: Option<reqwest::Response>,
    decoder: StreamDecoder,
    control: CallControl,
    // Boxing reqwest's Bytes retains its allocation without needing another
    // dependency or copying each network chunk into a Vec.
    chunk: Option<Box<dyn AsRef<[u8]> + Send>>,
    offset: usize,
    budget: usize,
}

pub(super) fn model_stream(
    response: reqwest::Response,
    decoder: StreamDecoder,
    control: CallControl,
) -> ModelStream {
    let state = State {
        response: Some(response),
        decoder,
        control,
        chunk: None,
        offset: 0,
        budget: 0,
    };
    Box::pin(
        futures_util::stream::unfold(Some(state), |state| async move {
            let mut state = state?;
            loop {
                if let Err(error) = check_control(&state.control) {
                    return Some((error_delta(error), None));
                }
                state.budget += 1;
                if state.budget >= PROGRESS_BUDGET {
                    state.budget = 0;
                    tokio::task::yield_now().await;
                    if let Err(error) = check_control(&state.control) {
                        return Some((error_delta(error), None));
                    }
                }
                let mut input: &[u8] = match &state.chunk {
                    Some(chunk) => {
                        let bytes = chunk.as_ref().as_ref();
                        &bytes[state.offset
                            ..bytes.len().min(state.offset.saturating_add(DECODE_WINDOW))]
                    }
                    None => &[],
                };
                let offered = input.len();
                let decoded = state.decoder.decode(&mut input);
                state.offset += offered - input.len();
                if let Err(error) = check_control(&state.control) {
                    return Some((error_delta(error), None));
                }
                match decoded {
                    Err(error) => return Some((error_delta(error), None)),
                    Ok(Some(delta)) => {
                        if matches!(delta, StreamDelta::Done { .. } | StreamDelta::Error { .. }) {
                            // Drop response and decoder now, including unconsumed
                            // tail bytes. Never wait for the remote EOF.
                            return Some((delta, None));
                        }
                        if state.decoder.is_terminal() {
                            state.response = None;
                            state.chunk = None;
                            state.offset = 0;
                        }
                        return Some((delta, Some(state)));
                    }
                    Ok(None) => {}
                }
                if let Some(chunk) = &state.chunk {
                    if state.offset < chunk.as_ref().as_ref().len() {
                        continue;
                    }
                    state.chunk = None;
                    state.offset = 0;
                }
                let Some(response) = state.response.as_mut() else {
                    return Some((
                        error_delta(ModelInvokeError::new(
                            ModelInvokeErrorKind::UnknownOutcome,
                            "stream decoder accepted terminal without delivering Done",
                        )),
                        None,
                    ));
                };
                match controlled(&state.control, response.chunk()).await {
                    Err(error) => return Some((error_delta(error), None)),
                    Ok(Err(error)) => return Some((error_delta(map_transport_error(error)), None)),
                    Ok(Ok(Some(chunk))) => state.chunk = Some(Box::new(chunk)),
                    Ok(Ok(None)) => {
                        let error = state.decoder.finish().err().unwrap_or_else(|| {
                            ModelInvokeError::new(
                                ModelInvokeErrorKind::UnknownOutcome,
                                "stream ended without a protocol terminal",
                            )
                        });
                        return Some((error_delta(error), None));
                    }
                }
            }
        })
        .fuse(),
    )
}

fn error_delta(error: ModelInvokeError) -> StreamDelta {
    StreamDelta::Error { error }
}
