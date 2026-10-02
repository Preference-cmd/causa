//! Public ports — the behavior seams external implementors fill in.
//!
//! Each trait here is transport-free and runtime-agnostic except for the
//! control planes' `CancellationToken` (re-exported at the crate root so the
//! port set is self-contained). The context model lives in `crate::context`.
//! A type belongs here iff it is the contract surface third parties
//! implement or call against; the kernel itself consumes none of it — the
//! canonical consumer (driver, executor) lives in `causa-runtime`. The
//! application persistence and material-selection policies live outside the
//! kernel. Context preparation exposes a shared exclusive material contract.

pub mod batch;
pub mod control;
pub mod gateway;
pub mod prepare;
pub mod source;
pub mod tool;
