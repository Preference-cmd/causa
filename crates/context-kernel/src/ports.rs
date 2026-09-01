//! Public ports — the behavior seams external implementors fill in.
//!
//! Each trait here is transport-free and runtime-agnostic except for the
//! control planes' `CancellationToken` (re-exported at the crate root so the
//! port set is self-contained). The context model lives in `crate::context`.
//! A type belongs here iff it is the contract surface third parties
//! implement or call against; the kernel itself consumes none of it — the
//! canonical consumer (driver, executor) lives in `reimagine-agent-runtime`.

pub mod budget;
pub mod control;
pub mod gateway;
pub mod store;
pub mod tool;
