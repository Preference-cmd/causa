use crate::translation::streaming::{
    StreamLimits,
    common::{Budget, index, permanent, string},
};
use causa_kernel::ModelInvokeError;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
pub(super) struct Item {
    pub(super) id: String,
    pub(super) kind: String,
    pub(super) call_id: Option<String>,
    pub(super) name: Option<String>,
    pub(super) call_index: Option<usize>,
    pub(super) arguments: String,
    pub(super) arguments_done: bool,
    pub(super) closed: bool,
    pub(super) parts: BTreeMap<usize, String>,
    pub(super) closed_parts: BTreeSet<usize>,
    pub(super) completed_parts: BTreeSet<usize>,
    pub(super) summaries: BTreeSet<usize>,
    pub(super) closed_summaries: BTreeSet<usize>,
    pub(super) completed_summaries: BTreeSet<usize>,
}

pub(in crate::translation::streaming) struct Responses {
    pub(super) budget: Budget,
    pub(super) id: Option<String>,
    pub(super) items: BTreeMap<usize, Item>,
    pub(super) tools: usize,
    pub(super) parts: usize,
}

impl Responses {
    pub fn new(limits: StreamLimits) -> Self {
        Self {
            budget: Budget::new(limits),
            id: None,
            items: BTreeMap::new(),
            tools: 0,
            parts: 0,
        }
    }
    pub(super) fn response_id(&mut self, response: &Value) -> Result<(), ModelInvokeError> {
        if let Some(id) = response.get("id") {
            let id = id
                .as_str()
                .ok_or_else(|| permanent("Responses invalid response id"))?;
            if self.id.as_deref().is_some_and(|previous| previous != id) {
                return Err(permanent("Responses response id changed"));
            }
            if self.id.is_none() {
                self.budget.add(id.len())?;
                self.id = Some(id.into());
            }
        }
        Ok(())
    }
    pub(super) fn item<'a>(
        items: &'a mut BTreeMap<usize, Item>,
        value: &Value,
    ) -> Result<&'a mut Item, ModelInvokeError> {
        let item = items
            .get_mut(&index(value, "output_index")?)
            .ok_or_else(|| permanent("Responses event for undeclared item"))?;
        if item.closed {
            return Err(permanent("Responses event for closed item"));
        }
        if string(value, "item_id")? != item.id {
            return Err(permanent("Responses item id/index conflict"));
        }
        Ok(item)
    }
    pub(super) fn identity(item: &Item, wire: &Value) -> Result<(), ModelInvokeError> {
        if string(wire, "id")? != item.id || string(wire, "type")? != item.kind {
            return Err(permanent("Responses output identity changed"));
        }
        if item.kind == "function_call"
            && (wire.get("call_id").and_then(Value::as_str) != item.call_id.as_deref()
                || wire.get("name").and_then(Value::as_str) != item.name.as_deref())
        {
            return Err(permanent("Responses tool identity changed"));
        }
        Ok(())
    }
    pub(super) fn arguments(args: &str) -> Result<(), ModelInvokeError> {
        serde_json::from_str::<Value>(args)
            .map_err(|_| permanent("Responses malformed or missing tool arguments"))?;
        Ok(())
    }
}
