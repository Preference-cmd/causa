use super::{SseEvent, SseParser};
use causa_kernel::ModelInvokeErrorKind;

fn events(chunks: &[&[u8]]) -> Vec<SseEvent> {
    let mut parser = SseParser::new();
    let mut out = Vec::new();
    for chunk in chunks {
        let mut input = *chunk;
        while let Some(event) = parser.decode(&mut input).unwrap() {
            out.push(event);
        }
        assert!(input.is_empty());
    }
    parser.finish().unwrap();
    out
}

#[test]
fn arbitrary_byte_boundaries_preserve_unicode_bom_and_all_line_endings() {
    let input = "\u{feff}: heartbeat\r\nevent: text\rdata: 你好🦀\r\ndata:\rdata: \\u4f60\n\ndata: tail\r\r".as_bytes();
    let expected = vec![
        SseEvent {
            event: Some("text".into()),
            data: "你好🦀\n\n\\u4f60".into(),
        },
        SseEvent {
            event: None,
            data: "tail".into(),
        },
    ];
    assert_eq!(events(&[input]), expected);
    for split in 0..=input.len() {
        assert_eq!(events(&[&input[..split], &input[split..]]), expected);
    }
    assert_eq!(events(&input.chunks(1).collect::<Vec<_>>()), expected);
}

#[test]
fn empty_fields_unknown_fields_and_comments_follow_sse_framing() {
    assert_eq!(
        events(&[b": comment\nid: 4\nretry: 10\n\ndata\ndata: one\nevent:\ndata:  two\n\n"]),
        vec![SseEvent {
            event: None,
            data: "\none\n two".into()
        }]
    );
}

#[test]
fn coalesced_events_are_not_mistaken_for_one_oversized_event() {
    let mut parser = SseParser::new().with_limits(7, 9);
    let bytes = b"data: x\n\ndata: y\n\ndata: z\n\n";
    let mut input = bytes.as_slice();
    for expected in ["x", "y", "z"] {
        assert_eq!(parser.decode(&mut input).unwrap().unwrap().data, expected);
    }
    assert!(input.is_empty());
    parser.finish().unwrap();
}

#[test]
fn oversized_line_multiline_event_and_ignored_fields_fail_closed() {
    for (line, event, bytes) in [
        (7, 100, b"data: xx\n\n".as_slice()),
        (7, 14, b"data: x\ndata: y\n\n".as_slice()),
        (20, 12, b":123456\n:123456\n\n".as_slice()),
    ] {
        let mut parser = SseParser::new().with_limits(line, event);
        let mut input = bytes;
        assert_eq!(
            parser.decode(&mut input).unwrap_err().kind,
            ModelInvokeErrorKind::Undecodable
        );
        assert!(parser.decode(&mut b"data: ok\n\n".as_slice()).is_err());
        assert!(parser.finish().is_err());
    }
}

#[test]
fn eof_cannot_dispatch_a_partial_event_or_replace_invalid_unicode() {
    for bytes in [b"data: x".as_slice(), b"data: x\n", b"event: end\n"] {
        let mut parser = SseParser::new();
        let mut input = bytes;
        assert!(parser.decode(&mut input).unwrap().is_none());
        assert_eq!(
            parser.finish().unwrap_err().kind,
            ModelInvokeErrorKind::UnknownOutcome
        );
    }
    for bytes in [b"data: \xff\n\n".as_slice(), b": \xf0\x9f"] {
        let mut parser = SseParser::new();
        let mut input = bytes;
        let result = parser.decode(&mut input);
        let error = match result {
            Err(error) => error,
            Ok(None) => parser.finish().unwrap_err(),
            Ok(Some(_)) => panic!("invalid UTF-8 produced an event"),
        };
        assert_eq!(error.kind, ModelInvokeErrorKind::Undecodable);
    }
}

#[test]
fn completed_prefix_is_returned_before_a_malformed_tail() {
    let mut parser = SseParser::new();
    let mut input = b"data: first\n\ndata: \xff\n\n".as_slice();
    assert_eq!(parser.decode(&mut input).unwrap().unwrap().data, "first");
    assert!(!input.is_empty());
    assert!(parser.decode(&mut input).is_err());
}

#[test]
fn successful_finish_is_idempotent_but_cannot_restart_input() {
    let mut parser = SseParser::new();
    parser.finish().unwrap();
    parser.finish().unwrap();
    assert!(parser.decode(&mut b"".as_slice()).unwrap().is_none());
    assert!(parser.decode(&mut b"data: x\n\n".as_slice()).is_err());
}
