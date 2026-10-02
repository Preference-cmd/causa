//! Private per-processor handoff checks used by whole-batch execution.

mod handoff;

pub(crate) use handoff::process_phase;
