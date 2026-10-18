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
