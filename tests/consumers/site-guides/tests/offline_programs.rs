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
fn context_editing_replaces_three_blocks_and_appends_one() {
    assert_eq!(run(env!("CARGO_BIN_EXE_context_editing")), "4 blocks\n");
}

#[test]
fn write_a_tool_reads_the_note_and_the_model_answers() {
    assert_eq!(
        run(env!("CARGO_BIN_EXE_write_a_tool")),
        "read_note: Succeeded {\"text\":\"Buy milk.\"}\nThe note says: \"Buy milk.\"\n"
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
