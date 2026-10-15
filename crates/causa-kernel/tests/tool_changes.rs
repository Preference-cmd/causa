//! Tool changes fold into the tool state at a position, and the change
//! between two sets folds back into the second set.

use causa_kernel::{
    BlockContent, BlockId, BlockMeta, ContextBlock, Role, ToolChange, ToolDefinition, current_tools,
};
use serde_json::json;
use std::sync::atomic::{AtomicU64, Ordering};

fn tool(name: &str, description: &str) -> ToolDefinition {
    ToolDefinition::new(name, description, json!({"type": "object"}))
}

fn block(content: BlockContent) -> ContextBlock {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let id = uuid::Uuid::from_u128(NEXT.fetch_add(1, Ordering::Relaxed).into());
    ContextBlock::new(BlockId::new(id), content, BlockMeta::default())
}

fn change(added: Vec<ToolDefinition>, removed: &[&str]) -> ContextBlock {
    block(BlockContent::ToolChange(ToolChange::new(
        added,
        removed.iter().map(|name| (*name).to_owned()).collect(),
    )))
}

fn names(tools: &[ToolDefinition]) -> Vec<&str> {
    tools.iter().map(|tool| tool.name.as_str()).collect()
}

#[test]
fn the_state_folds_every_change_in_order() {
    let blocks = vec![
        change(vec![tool("a", "A"), tool("b", "B")], &[]),
        block(BlockContent::text(Role::User, "between")),
        change(vec![tool("c", "C")], &["a"]),
        // A same-name addition replaces the definition in place.
        change(vec![tool("b", "B2")], &[]),
        // Withdrawing a name that is not offered changes nothing.
        change(vec![], &["missing"]),
    ];
    let tools = current_tools(&blocks);
    assert_eq!(names(&tools), ["b", "c"]);
    assert_eq!(tools[0].description, "B2");
    assert!(current_tools(&blocks[..1]).len() == 2);
    assert!(current_tools(&[]).is_empty());
}

#[test]
fn removal_applies_before_addition_within_one_change() {
    let blocks = vec![
        change(vec![tool("a", "A")], &[]),
        change(vec![tool("a", "A2")], &["a"]),
    ];
    let tools = current_tools(&blocks);
    assert_eq!(names(&tools), ["a"]);
    assert_eq!(tools[0].description, "A2");
}

#[test]
fn the_change_between_two_sets_folds_into_the_second() {
    let previous = vec![tool("a", "A"), tool("b", "B"), tool("c", "C")];
    let current = vec![tool("c", "C"), tool("b", "B2"), tool("d", "D")];
    let between = ToolChange::between(&previous, &current);
    assert_eq!(names(&between.added), ["b", "d"]);
    assert_eq!(between.removed, ["a", "b"]);

    let mut folded = previous.clone();
    between.apply_to(&mut folded);
    assert_eq!(
        ToolChange::between(&folded, &current),
        ToolChange::default()
    );
}

#[test]
fn reordering_is_not_a_change() {
    let previous = vec![tool("a", "A"), tool("b", "B")];
    let current = vec![tool("b", "B"), tool("a", "A")];
    assert!(ToolChange::between(&previous, &current).is_empty());
    assert!(!ToolChange::between(&[], &current).is_empty());
}
