//! ConversationStore port smoke test (Slice 5A Phase B).
//!
//! Pins the public error-surface shape (`Display` text, variant names)
//! so downstream hosts can pattern-match without surprises. The trait
//! itself needs no test beyond `cargo check`; implementors live in
//! `agent-runtime/examples/` and have their own integration tests.

use reimagine_context_kernel::{ConversationId, ConversationStoreError};

#[test]
fn error_display_messages_are_stable() {
    let cases = [
        (
            ConversationStoreError::NotFound("conv-x".into()),
            "conversation not found: conv-x",
        ),
        (ConversationStoreError::Io("eof".into()), "io error: eof"),
        (
            ConversationStoreError::Serialization("bad json".into()),
            "serialization error: bad json",
        ),
        (
            ConversationStoreError::Corrupted("truncated".into()),
            "corrupted data: truncated",
        ),
    ];
    for (err, expected) in cases {
        assert_eq!(format!("{}", err), expected);
    }
}

#[test]
fn conversation_id_is_passthrough() {
    // Smoke: the port trait works with any `ConversationId`;
    // the field is a `String`-backed newtype and `Eq`/`Hash`-stable.
    let a = ConversationId("a".into());
    let b = ConversationId("a".into());
    assert_eq!(a, b);
    assert_eq!(format!("{:?}", a), "ConversationId(\"a\")");
}
