//! Staged perimeter — reference driver, config axes, executor, fakes, noop
//! defaults. Everything here is deliberately root re-exported (see `lib.rs`)
//! but holds no claim on the kernel contract: canonical and port modules must
//! never reference items in this subtree (Slice 1.5 proposal, Phase C/D).

pub mod config;
pub mod defaults;
pub mod driver;
pub mod executor;
pub mod fakes;
