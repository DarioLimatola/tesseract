//! The engine's API towards tasks, runners and monitors.
//!
//! A model maps a context of tokens to the final state of the last token
//! and learns from a wanted output. The engine knows nothing about any task.

use crate::attention::{AttentionHead, AttentionParams};
use crate::entity::Entity;
use crate::memory::{MemoryBlock, MemoryParams};
use crate::state::{State, Token};
use crate::structure::{Census, StructurePolicy, StructureReport, UnitState};
use crate::unit::{Unit, UnitParams};

/// What one learning step did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepReport {
    /// Parameter bits that changed (connections revived or flipped).
    pub changes: u32,
    /// Units recruited.
    pub recruited: u32,
    /// True when the example could not be stored for lack of free units.
    pub dropped: bool,
}

/// A read-only view of one unit, for snapshots and the neuron view.
#[derive(Clone, Debug)]
pub struct UnitView {
    /// Life state.
    pub state: UnitState,
    /// Times the unit fired (or, for a head, selected a candidate).
    pub usage: u64,
    /// Live input connections.
    pub live: u32,
    /// Bias: the unit fires when `matches - mismatches >= bias`.
    pub bias: i32,
    /// Sign bits of the input connections.
    pub input_sign: Entity,
    /// Mask bits of the input connections.
    pub input_mask: Entity,
    /// Protected input connections.
    pub input_protected: Entity,
    /// Positions the unit votes up.
    pub output_plus: Entity,
    /// Positions the unit votes down.
    pub output_minus: Entity,
    /// Counters of the input connections, when the model is small enough
    /// to export them.
    pub counters: Option<Vec<i16>>,
}

/// A read-only view of one block.
#[derive(Clone, Debug)]
pub struct BlockView {
    /// Kind of block: `"unit"`, `"memory"` or `"attention"`.
    pub kind: &'static str,
    /// Its units. Free units are included without detail.
    pub units: Vec<UnitView>,
}

/// A trainable model built from the engine's parts.
pub trait Model: Send + Sync {
    /// Final state of the last token of `context`.
    fn forward(&self, context: &[Token]) -> State;

    /// Learns from one example: the context and the wanted bits of the
    /// output, at the positions selected by `target_mask`.
    fn learn(&mut self, context: &[Token], target: &Entity, target_mask: &Entity) -> StepReport;

    /// One step of the structure policy.
    fn structure_step(&mut self, policy: &StructurePolicy, error_rising: bool) -> StructureReport;

    /// Counts of units and connections.
    fn census(&self) -> Census;

    /// Views of the blocks, for snapshots.
    fn views(&self) -> Vec<BlockView>;

    /// Bytes of parameters read by the forward pass.
    fn parameter_bytes(&self) -> usize;
}

/// Index of the candidate closest to the output, comparing only positions
/// that are in `mask` and known. `None` when no such position exists or
/// there are no candidates. Ties go to the lowest index.
pub fn nearest_candidate(output: &Token, candidates: &[Entity], mask: &Entity) -> Option<usize> {
    let scope = *mask & output.known;
    if scope.is_zero() {
        return None;
    }
    let mut best: Option<(u32, usize)> = None;
    for (i, c) in candidates.iter().enumerate() {
        let s = output.bits.agreement(c, &scope);
        if best.is_none_or(|(bs, _)| s > bs) {
            best = Some((s, i));
        }
    }
    best.map(|(_, i)| i)
}

/// Number of units above which counters are left out of views.
const COUNTER_VIEW_LIMIT: usize = 64;

// ---------------------------------------------------------------------

/// One counting unit that votes its output bit into one position.
#[derive(Clone, Debug)]
pub struct UnitModel {
    /// The unit.
    pub unit: Unit,
    /// Its learning settings.
    pub params: UnitParams,
    /// Position of the state that receives the output: `+1` when the unit
    /// fires, `-1` when it does not.
    pub output_position: usize,
}

impl UnitModel {
    /// A unit with no live connection, reading every position except its
    /// output position.
    pub fn new(params: UnitParams, output_position: usize) -> Self {
        let mut unit = Unit::empty();
        unit.allowed.set(output_position, false);
        Self {
            unit,
            params,
            output_position,
        }
    }
}

impl Model for UnitModel {
    fn forward(&self, context: &[Token]) -> State {
        let Some(query) = context.last() else {
            return State::zero();
        };
        let mut state = State::from_token(query, 1);
        let bit = Entity::bit_mask(self.output_position);
        if self.unit.output(&query.bits) {
            state.add_votes(&bit, &Entity::ZERO);
        } else {
            state.add_votes(&Entity::ZERO, &bit);
        }
        state
    }

    fn learn(&mut self, context: &[Token], target: &Entity, target_mask: &Entity) -> StepReport {
        let mut report = StepReport::default();
        let Some(query) = context.last() else {
            return report;
        };
        if !target_mask.get(self.output_position) {
            return report;
        }
        let desired = target.get(self.output_position);
        let update = self.unit.learn(&query.bits, desired, &self.params);
        report.changes = update.growth.changes() + u32::from(update.bias_moved);
        report
    }

    fn structure_step(&mut self, policy: &StructurePolicy, error_rising: bool) -> StructureReport {
        let mut report = StructureReport::default();
        if error_rising {
            report.skipped = true;
        } else {
            report.connections_pruned = self.unit.prune(
                &self.params,
                policy.connection_budget,
                policy.min_live_inputs,
            );
        }
        report
    }

    fn census(&self) -> Census {
        let mut c = Census {
            units_active: 1,
            ..Census::default()
        };
        crate::structure::count_connections(
            &mut c,
            &self.unit.mask,
            &self.unit.protected,
            &self.unit.counters,
            true,
        );
        c
    }

    fn views(&self) -> Vec<BlockView> {
        vec![BlockView {
            kind: "unit",
            units: vec![UnitView {
                state: UnitState::Active,
                usage: self.unit.usage,
                live: self.unit.live,
                bias: self.unit.bias,
                input_sign: self.unit.sign,
                input_mask: self.unit.mask,
                input_protected: self.unit.protected,
                output_plus: Entity::bit_mask(self.output_position),
                output_minus: Entity::ZERO,
                counters: Some(self.unit.counters.clone()),
            }],
        }]
    }

    fn parameter_bytes(&self) -> usize {
        2 * std::mem::size_of::<Entity>() + 8
    }
}

// ---------------------------------------------------------------------

/// One memory block applied to the last token.
#[derive(Clone, Debug)]
pub struct MemoryModel {
    /// The block.
    pub block: MemoryBlock,
}

impl MemoryModel {
    /// A block of free units.
    pub fn new(units: usize, params: MemoryParams) -> Self {
        Self {
            block: MemoryBlock::new(units, params),
        }
    }
}

impl Model for MemoryModel {
    fn forward(&self, context: &[Token]) -> State {
        let Some(query) = context.last() else {
            return State::zero();
        };
        let mut state = State::from_token(query, 1);
        self.block.forward(query, &mut state);
        state
    }

    fn learn(&mut self, context: &[Token], target: &Entity, target_mask: &Entity) -> StepReport {
        let Some(query) = context.last() else {
            return StepReport::default();
        };
        let state = State::from_token(query, 1);
        let update = self.block.learn(query, &state, target, target_mask);
        StepReport {
            changes: update.growth.changes(),
            recruited: update.recruited,
            dropped: update.dropped,
        }
    }

    fn structure_step(&mut self, policy: &StructurePolicy, error_rising: bool) -> StructureReport {
        self.block.structure_step(policy, error_rising)
    }

    fn census(&self) -> Census {
        self.block.census()
    }

    fn views(&self) -> Vec<BlockView> {
        let b = &self.block;
        let with_counters = b.units() <= COUNTER_VIEW_LIMIT;
        let units = (0..b.units())
            .map(|u| {
                let (sign, mask) = b.unit_address(u);
                let (plus, minus) = b.unit_value(u);
                UnitView {
                    state: b.unit_state(u),
                    usage: b.unit_usage(u),
                    live: b.unit_live(u),
                    bias: b.unit_bias(u),
                    input_sign: *sign,
                    input_mask: *mask,
                    input_protected: Entity::ZERO,
                    output_plus: *plus,
                    output_minus: *minus,
                    counters: (with_counters && b.unit_state(u) != UnitState::Free)
                        .then(|| b.unit_address_counters(u).to_vec()),
                }
            })
            .collect();
        vec![BlockView {
            kind: "memory",
            units,
        }]
    }

    fn parameter_bytes(&self) -> usize {
        self.block.units() * (4 * std::mem::size_of::<Entity>() + 8)
    }
}

// ---------------------------------------------------------------------

/// One attention head over the context.
#[derive(Clone, Debug)]
pub struct AttentionModel {
    /// The head.
    pub head: AttentionHead,
}

impl AttentionModel {
    /// A head that matches on everything and writes nothing.
    pub fn new(params: AttentionParams) -> Self {
        Self {
            head: AttentionHead::new(params),
        }
    }
}

impl Model for AttentionModel {
    fn forward(&self, context: &[Token]) -> State {
        self.head.forward(context)
    }

    fn learn(&mut self, context: &[Token], target: &Entity, target_mask: &Entity) -> StepReport {
        let update = self.head.learn(context, target, target_mask);
        StepReport {
            changes: update.growth.changes(),
            ..StepReport::default()
        }
    }

    fn structure_step(&mut self, policy: &StructurePolicy, error_rising: bool) -> StructureReport {
        self.head.structure_step(policy, error_rising)
    }

    fn census(&self) -> Census {
        self.head.census()
    }

    fn views(&self) -> Vec<BlockView> {
        vec![BlockView {
            kind: "attention",
            units: vec![UnitView {
                state: UnitState::Active,
                usage: self.head.usage(),
                live: self.head.match_mask().count_ones(),
                bias: 0,
                input_sign: Entity::ONES,
                input_mask: *self.head.match_mask(),
                input_protected: *self.head.match_protected(),
                output_plus: *self.head.value_mask(),
                output_minus: Entity::ZERO,
                counters: Some(self.head.match_counters().to_vec()),
            }],
        }]
    }

    fn parameter_bytes(&self) -> usize {
        2 * std::mem::size_of::<Entity>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::DOMAINS;
    use crate::rng::Rng;

    #[test]
    fn nearest_candidate_needs_known_positions() {
        let mut rng = Rng::new(51);
        let candidates: Vec<Entity> = (0..4).map(|_| rng.entity()).collect();
        // Two different domains, whatever the geometry.
        let mask = Entity::domain_mask(DOMAINS - 1);
        let unknown = Token::partial(candidates[1], Entity::domain_mask(0));
        assert_eq!(nearest_candidate(&unknown, &candidates, &mask), None);
        let known = Token::full(candidates[2]);
        assert_eq!(nearest_candidate(&known, &candidates, &mask), Some(2));
        assert_eq!(nearest_candidate(&known, &[], &mask), None);
    }

    #[test]
    fn unit_model_votes_its_output_bit() {
        let model = UnitModel::new(UnitParams::default(), 4095);
        let x = Token::partial(Rng::new(52).entity(), !Entity::bit_mask(4095));
        // An empty unit with bias 0 fires.
        let out = model.forward(&[x]).token();
        assert!(out.bits.get(4095) && out.known.get(4095));
    }
}
