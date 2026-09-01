//! Public ports — the behavior seams external implementors fill in.
//!
//! Each trait here is transport-free and runtime-agnostic except for the
//! control planes' `CancellationToken` (re-exported at the crate root so the
//! port set is self-contained). The context model lives in `crate::context`; the staged
//! reference implementation that consumes these ports lives in
//! `crate::internal` and holds no claim on this contract.

pub mod budget;
pub mod control;
pub mod gateway;
pub mod store;
pub mod tool;
