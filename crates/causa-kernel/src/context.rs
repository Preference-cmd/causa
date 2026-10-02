//! Material vocabulary and atomic context edits.
//!
//! Context contains ordered blocks; request frames select independent material.
//! Execution identity and behavior belong to consumers of the kernel ports.

pub mod block;
pub mod ids;
pub mod material;
pub mod model;
pub mod tool_data;
