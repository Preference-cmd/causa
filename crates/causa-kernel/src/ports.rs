//! Public ports — the behavior seams external implementors fill in.
//!
//! Each trait here is transport-free and runtime-agnostic except for the
//! control planes' `CancellationToken` (re-exported at the crate root so the
//! port set is self-contained). The context model lives in `crate::context`.
//! A type belongs here iff it is the contract surface third parties
//! implement or call against; the kernel itself consumes none of it — the
//! canonical consumer (driver, executor) lives in `causa-runtime`. The
//! conversation-persistence, budget/compaction, and host↔driver interaction
//! seams are the reference harness's opinions, not cross-harness
//! capabilities, and live in `causa-runtime`.

pub mod control;
pub mod gateway;
pub mod source;
pub mod tool;
