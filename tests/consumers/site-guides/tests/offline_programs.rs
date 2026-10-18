//! Runs the site's offline programs and checks what they print.

use std::process::Command;

fn run(program: &str) -> String {
    let output = Command::new(program).output().expect("program starts");
    assert!(
        output.status.success(),
        "{program} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("UTF-8 output")
}

#[test]
fn context_editing_replaces_three_blocks_and_refuses_a_reused_id() {
    assert_eq!(
        run(env!("CARGO_BIN_EXE_context_editing")),
        "System: Answer briefly.\n\
         System: Earlier: the user is building a Rust agent that reads notes, and likes short answers.\n\
         User: Which crate should I use for HTTP?\nSystem: Name one crate.\n3 blocks replaced\n\
         refused (id reused for other content): 1 block handed back, 4 blocks before and after\n"
    );
}

#[test]
fn write_a_tool_reads_the_note_and_the_model_answers() {
    assert_eq!(
        run(env!("CARGO_BIN_EXE_write_a_tool")),
        "read_note: Succeeded {\"text\":\"Buy milk.\"}\nThe note says: Buy milk.\n"
    );
}

#[test]
fn connect_mcp_servers_lists_and_calls_the_in_process_tool() {
    assert_eq!(
        run(env!("CARGO_BIN_EXE_connect_mcp_servers")),
        "tool: mcp_notes_read_note\nresult: Succeeded \"Buy milk.\"\nThe note says: Buy milk.\n"
    );
}

#[test]
fn choose_a_provider_sends_one_context_over_three_protocols() {
    assert_eq!(
        run(env!("CARGO_BIN_EXE_choose_a_provider")),
        "/v1/messages x-api-key max_tokens=200 -> Hello from Anthropic Messages.\n\
         /v1/chat/completions authorization max_tokens=200 -> Hello from Chat Completions.\n\
         /v1/responses authorization max_output_tokens=200 -> Hello from Responses.\n"
    );
}

#[test]
fn save_and_restore_continues_the_saved_conversation() {
    assert_eq!(
        run(env!("CARGO_BIN_EXE_save_and_restore")),
        "Message 1 received. (2 blocks saved)\nMessage 2 received. (4 blocks saved)\n"
    );
}

#[test]
fn prepare_each_request_sends_the_last_two_turns() {
    assert_eq!(
        run(env!("CARGO_BIN_EXE_prepare_each_request")),
        "System: Answer in one word.\nToolChange: +lookup\nUser: Second?\nAssistant: Two.\nUser: Third?\n\
         The context holds 8 blocks.\n"
    );
}

#[test]
fn process_tool_calls_asks_and_keeps_declaration_order() {
    assert_eq!(
        run(env!("CARGO_BIN_EXE_process_tool_calls")),
        "allow read_note todo.txt? true\nallow read_note secret.txt? false\n\
         Succeeded {\"text\":\"Buy milk.\"}\n\
         Rejected {\"error\":\"the user did not allow this call\"}\n\
         I could read one of the two notes.\n"
    );
}

#[test]
fn media_renders_per_protocol() {
    let placeholder = "[media: application/pdf report.pdf sha256-82952bb9e32344c6]";
    assert_eq!(
        run(env!("CARGO_BIN_EXE_media")),
        format!(
            "Anthropic Messages:\n  user[text]\n  assistant[tool_use]\n  user[tool_result[text, image]]\n  \
             assistant[text]\n  user[text, document]\n\
             Chat Completions:\n  user[text]\n  assistant\n  tool[text]\n  user[text, image_url]\n  \
             assistant[text]\n  user[text, text {placeholder}]\n\
             Responses:\n  user[input_text]\n  function_call\n  function_call_output[input_text, input_image]\n  \
             assistant[output_text]\n  user[input_text, input_text {placeholder}]\n"
        )
    );
}

#[test]
fn retries_and_observation_retries_once_and_reports_each_step() {
    assert_eq!(
        run(env!("CARGO_BIN_EXE_retries_and_observation")),
        "committed: 1 blocks\nrequest: 2 blocks\n  provider: call 1 refused (429)\n  \
         provider: call 2 answered\noutput: ToolUse\ncommitted: 1 blocks\ntools: 1 calls\n\
         tools: 1 results\ncommitted: 1 blocks\nrequest: 4 blocks\n  provider: call 3 answered\n\
         output: EndTurn\ncommitted: 1 blocks\nIt is 09:30.\n"
    );
}
