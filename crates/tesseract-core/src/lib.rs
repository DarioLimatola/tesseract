//! Tesseract core: a binary, vectorised, multi-dimensional learning engine.
//!
//! Principle: count and compare, never multiply. Integers are produced by
//! counting bits and consumed by comparisons that give bits back. There is
//! no floating point in this crate.
//!
//! - [`entity`]: the 8^4-bit entity and its axis operations.
//! - [`code`]: codes for a dimension (thermometer, category).
//! - [`state`]: the integer state of a token.
//! - [`counters`]: hidden learning counters and the ternary automaton.
//! - [`unit`]: the ternary counting unit and its learning rule.
//! - [`memory`]: the memory block (match-and-count over learned addresses).
//! - [`attention`]: the attention head (match-and-count over the context).
//! - [`structure`]: the structure policy (gentle pruning, dormant state).
//! - [`model`]: the API used by tasks, runners and monitors.
//! - [`rng`]: deterministic random bits.
//!
//! This crate knows nothing about tasks, files or user interfaces.

pub mod attention;
pub mod code;
pub mod counters;
pub mod entity;
pub mod memory;
pub mod model;
pub mod rng;
pub mod state;
pub mod structure;
pub mod unit;

pub use attention::{AttentionHead, AttentionParams};
pub use counters::Automaton;
pub use entity::{Axis, BITS, DOMAIN_BITS, DOMAINS, Entity, SIDE, VALUES_PER_DOMAIN, WORDS};
pub use memory::{MemoryBlock, MemoryParams};
pub use model::{
    AttentionModel, BlockView, MemoryModel, Model, StepReport, UnitModel, UnitView,
    nearest_candidate,
};
pub use rng::Rng;
pub use state::{STATE_LIMIT, State, Token};
pub use structure::{Census, StructurePolicy, StructureReport, UnitState};
pub use unit::{Unit, UnitParams};

/// Version of the engine, recorded in every run folder.
pub const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");
