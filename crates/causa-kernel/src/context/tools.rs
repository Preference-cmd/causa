//! Tool declaration facts: which tools the model was offered, and from where
//! in the conversation.
//!
//! A [`ToolChange`] block records that the offered set changed at its
//! position. The tool state at any position is the fold of every earlier
//! change, so the context alone says which tools the model saw when, and a
//! renderer can keep the request-level tool list fixed while the set changes.
//! Execution vocabulary (the `Tool` trait, the bound `ToolSurface`) lives in
//! `crate::ports::tool`.

use serde::{Deserialize, Serialize};

use crate::context::block::{BlockContent, ContextBlock};

/// The model-facing description of one callable tool; renderers map it onto
/// the protocol's tool entries (name, description, parameter schema).
///
/// Part of the stable serialized set, because a [`ToolChange`] holds it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolDefinition {
    /// The name the model uses to invoke the tool.
    pub name: String,
    /// Short summary of what the tool does, shown to the model.
    pub description: String,
    /// Schema of the arguments the tool accepts, embedded verbatim in the
    /// rendered request.
    pub parameters: serde_json::Value,
}
impl ToolDefinition {
    /// A definition from its name, description and JSON Schema parameters.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

/// A change to the tools offered to the model, effective from the position
/// of its block onward.
///
/// Applying a change removes the `removed` names first, then adds each of
/// `added`, a definition replacing any current one of the same name. The
/// first change in a context declares the initial set; [`current_tools`]
/// folds them all.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolChange {
    /// Definitions offered from here on, replacing same-name definitions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub added: Vec<ToolDefinition>,
    /// Names of tools withdrawn from here on. A name that is not offered is
    /// ignored.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub removed: Vec<String>,
}
impl ToolChange {
    /// A change adding `added` after withdrawing `removed`.
    pub fn new(added: Vec<ToolDefinition>, removed: Vec<String>) -> Self {
        Self { added, removed }
    }

    /// The change that turns the set `previous` into the set `current`.
    ///
    /// A tool whose definition differs is withdrawn and added again; a tool
    /// that only moved in the list is unchanged, since order carries no
    /// meaning to the model. `added` follows the order of `current`,
    /// `removed` the order of `previous`.
    pub fn between(previous: &[ToolDefinition], current: &[ToolDefinition]) -> Self {
        let find = |set: &[ToolDefinition], name: &str| {
            set.iter()
                .find(|definition| definition.name == name)
                .cloned()
        };
        let added = current
            .iter()
            .filter(|definition| find(previous, &definition.name).as_ref() != Some(*definition))
            .cloned()
            .collect();
        let removed = previous
            .iter()
            .filter(|definition| find(current, &definition.name).as_ref() != Some(*definition))
            .map(|definition| definition.name.clone())
            .collect();
        Self { added, removed }
    }

    /// Whether the change adds and withdraws nothing.
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty()
    }

    /// Applies this change to a tool state in place.
    pub fn apply_to(&self, tools: &mut Vec<ToolDefinition>) {
        tools.retain(|definition| !self.removed.contains(&definition.name));
        for definition in &self.added {
            match tools
                .iter_mut()
                .find(|current| current.name == definition.name)
            {
                Some(current) => *current = definition.clone(),
                None => tools.push(definition.clone()),
            }
        }
    }
}

/// The tools offered after every [`ToolChange`] in `blocks`, in the order
/// they were first added.
pub fn current_tools(blocks: &[ContextBlock]) -> Vec<ToolDefinition> {
    let mut tools = Vec::new();
    for block in blocks {
        if let BlockContent::ToolChange(change) = block.content() {
            change.apply_to(&mut tools);
        }
    }
    tools
}
