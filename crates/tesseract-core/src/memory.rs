//! The memory block: match-and-count over learned addresses.
//!
//! A layer of ternary counting units. Each unit that fires adds its ternary
//! value vector to the state as votes. A block whose units have empty value
//! vectors writes nothing, so a new block is the identity.
//!
//! Parameters (signs, masks, biases) and learning counters live in separate
//! arrays: the forward pass touches parameters only.

use crate::counters::{Automaton, Growth, accumulate, grow, prune};
use crate::entity::{BITS, Entity};
use crate::state::{State, Token};
use crate::structure::{Census, StructurePolicy, StructureReport, UnitState, count_connections};
use crate::unit::{UnitParams, activation, match_count};

/// Settings of a memory block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryParams {
    /// Learning of the address connections.
    pub address: UnitParams,
    /// Automaton of the value connections.
    pub value: Automaton,
    /// A recruited unit fires when at least this share (per mille) of its
    /// live inputs agree with the input.
    pub vigilance_per_mille: u32,
    /// The state must agree with the target by at least this much.
    pub output_margin: i8,
    /// A unit that fired is blamed (its address learns "do not fire here")
    /// when it voted against more than this share (per mille) of the
    /// positions still in error.
    pub blame_per_mille: u32,
    /// Positions an address may connect to.
    pub input_mask: Entity,
}

impl Default for MemoryParams {
    fn default() -> Self {
        Self {
            address: UnitParams::default(),
            value: Automaton {
                live_at: 1,
                dormant_below: 1,
                limit: 15,
                protect_at: 16,
                signed: true,
            },
            vigilance_per_mille: 625,
            output_margin: 1,
            blame_per_mille: 250,
            input_mask: Entity::ONES,
        }
    }
}

/// What one learning step did to a block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlockUpdate {
    /// Units that fired.
    pub fired: u32,
    /// Free units committed to a new address.
    pub recruited: u32,
    /// Dormant units woken up.
    pub revived_units: u32,
    /// Fired units whose address was told not to fire here.
    pub blamed: u32,
    /// True when an example needed a new unit and none was free.
    pub dropped: bool,
    /// Connection changes (addresses and values).
    pub growth: Growth,
}

/// A layer of ternary counting units with value vectors.
#[derive(Clone, Debug)]
pub struct MemoryBlock {
    params: MemoryParams,
    // Parameters, read by the forward pass.
    addr_sign: Vec<Entity>,
    addr_mask: Vec<Entity>,
    live: Vec<u32>,
    bias: Vec<i32>,
    value_plus: Vec<Entity>,
    value_minus: Vec<Entity>,
    state: Vec<UnitState>,
    // Structure state.
    addr_protected: Vec<Entity>,
    value_protected: Vec<Entity>,
    pinned: Vec<bool>,
    usage: Vec<u64>,
    recent: Vec<u32>,
    idle: Vec<u32>,
    // Learning counters.
    addr_counters: Vec<i16>,
    value_counters: Vec<i16>,
    bias_counters: Vec<i16>,
    next_free: usize,
    prune_cursor: usize,
}

impl MemoryBlock {
    /// A block of `units` free units. It writes nothing until it learns.
    pub fn new(units: usize, params: MemoryParams) -> Self {
        Self {
            params,
            addr_sign: vec![Entity::ZERO; units],
            addr_mask: vec![Entity::ZERO; units],
            live: vec![0; units],
            bias: vec![0; units],
            value_plus: vec![Entity::ZERO; units],
            value_minus: vec![Entity::ZERO; units],
            state: vec![UnitState::Free; units],
            addr_protected: vec![Entity::ZERO; units],
            value_protected: vec![Entity::ZERO; units],
            pinned: vec![false; units],
            usage: vec![0; units],
            recent: vec![0; units],
            idle: vec![0; units],
            addr_counters: vec![0; units * BITS],
            value_counters: vec![0; units * BITS],
            bias_counters: vec![0; units],
            next_free: 0,
            prune_cursor: 0,
        }
    }

    /// Number of units (free, active and dormant).
    pub fn units(&self) -> usize {
        self.state.len()
    }

    /// Settings of the block.
    pub fn params(&self) -> &MemoryParams {
        &self.params
    }

    /// Life state of a unit.
    pub fn unit_state(&self, u: usize) -> UnitState {
        self.state[u]
    }

    /// Times a unit fired.
    pub fn unit_usage(&self, u: usize) -> u64 {
        self.usage[u]
    }

    /// Number of live input connections of a unit.
    pub fn unit_live(&self, u: usize) -> u32 {
        self.live[u]
    }

    /// Bias of a unit.
    pub fn unit_bias(&self, u: usize) -> i32 {
        self.bias[u]
    }

    /// Address of a unit: `(sign, mask)`.
    pub fn unit_address(&self, u: usize) -> (&Entity, &Entity) {
        (&self.addr_sign[u], &self.addr_mask[u])
    }

    /// Value vector of a unit: `(votes +1, votes -1)`.
    pub fn unit_value(&self, u: usize) -> (&Entity, &Entity) {
        (&self.value_plus[u], &self.value_minus[u])
    }

    /// Address counters of a unit.
    pub fn unit_address_counters(&self, u: usize) -> &[i16] {
        &self.addr_counters[u * BITS..(u + 1) * BITS]
    }

    /// Pins a unit: it is never made dormant by the policy.
    pub fn pin(&mut self, u: usize) {
        self.pinned[u] = true;
    }

    #[inline]
    fn unit_fires(&self, u: usize, x: &Entity) -> bool {
        let count = match_count(x, &self.addr_sign[u], &self.addr_mask[u]);
        activation(count, self.live[u], self.bias[u]) >= 0
    }

    /// Indices of the active units that fire on `x`, in increasing order.
    pub fn fired(&self, x: &Entity, out: &mut Vec<u32>) {
        out.clear();
        for u in 0..self.units() {
            if self.state[u] == UnitState::Active && self.unit_fires(u, x) {
                out.push(u as u32);
            }
        }
    }

    /// Adds the value vectors of the given units to the state.
    pub fn write(&self, fired: &[u32], state: &mut State) {
        for &u in fired {
            state.add_votes(&self.value_plus[u as usize], &self.value_minus[u as usize]);
        }
    }

    /// Forward pass for one token: reads `input`, writes votes into `state`.
    /// Returns the number of units that fired.
    pub fn forward(&self, input: &Token, state: &mut State) -> usize {
        let mut fired = Vec::new();
        self.fired(&input.bits, &mut fired);
        self.write(&fired, state);
        fired.len()
    }

    /// Forward pass for a batch of independent tokens. Each token starts
    /// from its own initial state. With the `threads` feature the batch is
    /// split across threads; results are identical for any thread count.
    pub fn forward_batch(&self, inputs: &[Token]) -> Vec<State> {
        let one = |t: &Token| {
            let mut s = State::from_token(t, 1);
            self.forward(t, &mut s);
            s
        };
        #[cfg(feature = "threads")]
        {
            use rayon::prelude::*;
            inputs.par_iter().map(one).collect()
        }
        #[cfg(not(feature = "threads"))]
        {
            inputs.iter().map(one).collect()
        }
    }

    /// Error-driven learning from one example.
    ///
    /// `input` is the token the addresses read, `state` the state before
    /// this block, `target` and `target_mask` the wanted bits of the state
    /// after it.
    ///
    /// 1. Units that fire and vote with the target get value evidence.
    /// 2. Units that fire and vote against it are blamed: their address
    ///    learns not to fire on this input.
    /// 3. When no unit helped, a dormant unit that matches is revived, or a
    ///    free unit is recruited with this input as its address.
    pub fn learn(
        &mut self,
        input: &Token,
        state: &State,
        target: &Entity,
        target_mask: &Entity,
    ) -> BlockUpdate {
        let mut update = BlockUpdate::default();
        let x = input.bits;
        let mut fired = Vec::new();
        self.fired(&x, &mut fired);
        update.fired = fired.len() as u32;
        let mut after = state.clone();
        self.write(&fired, &mut after);
        for &u in &fired {
            self.usage[u as usize] += 1;
            self.recent[u as usize] = self.recent[u as usize].saturating_add(1);
        }
        let (up, down) = after.error(target, target_mask, self.params.output_margin);
        let wanted = (up | down).count_ones();
        if wanted == 0 {
            return update;
        }

        let mut helped = false;
        for &u in &fired {
            let u = u as usize;
            let (vp, vm) = (self.value_plus[u], self.value_minus[u]);
            let hurts = (vp & down).count_ones() + (vm & up).count_ones();
            if u64::from(hurts) * 1000 > u64::from(self.params.blame_per_mille) * u64::from(wanted)
            {
                update.blamed += 1;
                update.growth.add(self.learn_address(u, &x, false));
            } else {
                helped = true;
                update.growth.add(self.learn_value(u, &up, &down));
            }
        }
        if helped {
            return update;
        }

        // Nothing useful fired: revive a matching dormant unit, or recruit.
        let dormant = (0..self.units())
            .find(|&u| self.state[u] == UnitState::Dormant && self.unit_fires(u, &x));
        if let Some(u) = dormant {
            self.state[u] = UnitState::Active;
            self.idle[u] = 0;
            self.recent[u] = 1;
            update.revived_units = 1;
            update.growth.add(self.learn_value(u, &up, &down));
            return update;
        }
        while self.next_free < self.units() && self.state[self.next_free] != UnitState::Free {
            self.next_free += 1;
        }
        if self.next_free >= self.units() {
            update.dropped = true;
            return update;
        }
        let u = self.next_free;
        update.growth.add(self.recruit(u, input, &up, &down));
        update.recruited = 1;
        update
    }

    /// Commits a free unit: its address becomes the known bits of `input`
    /// and its value the wanted change of the state.
    fn recruit(&mut self, u: usize, input: &Token, up: &Entity, down: &Entity) -> Growth {
        let a = self.params.address.automaton;
        let known = input.known & self.params.input_mask;
        let counters = &mut self.addr_counters[u * BITS..(u + 1) * BITS];
        for (p, c) in counters.iter_mut().enumerate() {
            *c = if !known.get(p) {
                0
            } else if input.bits.get(p) {
                a.live_at
            } else {
                -a.live_at
            };
        }
        let mut growth = grow(
            counters,
            &mut self.addr_sign[u],
            &mut self.addr_mask[u],
            &mut self.addr_protected[u],
            &known,
            &a,
        );
        let live = self.addr_mask[u].count_ones();
        self.live[u] = live;
        // Fire when `vigilance` of the live inputs agree:
        // matches - mismatches >= live * (2 * vigilance - 1).
        let v = i64::from(self.params.vigilance_per_mille.min(1000));
        self.bias[u] = ((i64::from(live) * (2 * v - 1000)) / 1000) as i32;
        self.bias_counters[u] = 0;
        self.state[u] = UnitState::Active;
        self.usage[u] = 1;
        self.recent[u] = 1;
        self.idle[u] = 0;

        let va = self.params.value;
        let counters = &mut self.value_counters[u * BITS..(u + 1) * BITS];
        for (p, c) in counters.iter_mut().enumerate() {
            *c = if up.get(p) {
                va.live_at
            } else if down.get(p) {
                -va.live_at
            } else {
                0
            };
        }
        growth.add(self.refresh_value(u, &(*up | *down)));
        growth
    }

    /// Gives a unit's value counters one unit of evidence and updates its
    /// value vector.
    fn learn_value(&mut self, u: usize, up: &Entity, down: &Entity) -> Growth {
        let limit = self.params.value.limit;
        accumulate(
            &mut self.value_counters[u * BITS..(u + 1) * BITS],
            up,
            down,
            limit,
        );
        self.refresh_value(u, &(*up | *down))
    }

    fn refresh_value(&mut self, u: usize, touched: &Entity) -> Growth {
        let mut sign = self.value_plus[u];
        let mut mask = self.value_plus[u] | self.value_minus[u];
        let growth = grow(
            &self.value_counters[u * BITS..(u + 1) * BITS],
            &mut sign,
            &mut mask,
            &mut self.value_protected[u],
            touched,
            &self.params.value,
        );
        self.value_plus[u] = sign & mask;
        self.value_minus[u] = mask.and_not(&sign);
        growth
    }

    /// Address learning of one unit, the same rule as [`crate::unit::Unit`].
    fn learn_address(&mut self, u: usize, x: &Entity, desired: bool) -> Growth {
        let p = self.params.address;
        let allowed = self.params.input_mask;
        let (plus, minus) = if desired {
            (*x & allowed, allowed.and_not(x))
        } else {
            (allowed.and_not(x), *x & allowed)
        };
        let counters = &mut self.addr_counters[u * BITS..(u + 1) * BITS];
        accumulate(counters, &plus, &minus, p.automaton.limit);
        let growth = grow(
            counters,
            &mut self.addr_sign[u],
            &mut self.addr_mask[u],
            &mut self.addr_protected[u],
            &allowed,
            &p.automaton,
        );
        self.live[u] = self.addr_mask[u].count_ones();
        self.bias_counters[u] += if desired { -1 } else { 1 };
        if self.bias_counters[u].abs() >= p.bias_step {
            self.bias[u] += i32::from(self.bias_counters[u].signum());
            self.bias_counters[u] = 0;
        }
        growth
    }

    /// One structure step. Does nothing while the error is rising.
    pub fn structure_step(
        &mut self,
        policy: &StructurePolicy,
        error_rising: bool,
    ) -> StructureReport {
        let mut report = StructureReport::default();
        if error_rising {
            report.skipped = true;
            for r in &mut self.recent {
                *r = 0;
            }
            return report;
        }
        let n = self.units();
        // Connections: weak, unprotected ones become dormant, on a budget
        // shared by the block. The cursor spreads the budget over units.
        let mut budget = policy.connection_budget;
        for k in 0..n {
            if budget == 0 {
                break;
            }
            let u = (self.prune_cursor + k) % n;
            if self.state[u] != UnitState::Active {
                continue;
            }
            let pruned = prune(
                &self.addr_counters[u * BITS..(u + 1) * BITS],
                &mut self.addr_mask[u],
                &self.addr_protected[u],
                &self.params.address.automaton,
                budget,
                policy.min_live_inputs,
            );
            if pruned > 0 {
                self.live[u] = self.addr_mask[u].count_ones();
                budget -= pruned;
                report.connections_pruned += pruned;
            }
        }
        if n > 0 {
            self.prune_cursor = (self.prune_cursor + 1) % n;
        }
        // Units: those idle for long enough become dormant, on a budget.
        let mut active = self
            .state
            .iter()
            .filter(|s| **s == UnitState::Active)
            .count() as u32;
        for u in 0..n {
            if self.state[u] != UnitState::Active {
                continue;
            }
            if self.recent[u] > 0 {
                self.idle[u] = 0;
            } else {
                self.idle[u] = self.idle[u].saturating_add(1);
            }
            self.recent[u] = 0;
            if policy.idle_steps_to_dormant > 0
                && self.idle[u] >= policy.idle_steps_to_dormant
                && !self.pinned[u]
                && report.units_pruned < policy.unit_budget
                && active > policy.min_live_units
            {
                self.state[u] = UnitState::Dormant;
                active -= 1;
                report.units_pruned += 1;
            }
        }
        report
    }

    /// Counts of units and connections.
    pub fn census(&self) -> Census {
        let mut c = Census::default();
        for u in 0..self.units() {
            match self.state[u] {
                UnitState::Free => {
                    c.units_free += 1;
                    continue;
                }
                UnitState::Active => c.units_active += 1,
                UnitState::Dormant => c.units_dormant += 1,
            }
            let active = self.state[u] == UnitState::Active;
            count_connections(
                &mut c,
                &self.addr_mask[u],
                &self.addr_protected[u],
                &self.addr_counters[u * BITS..(u + 1) * BITS],
                active,
            );
            count_connections(
                &mut c,
                &(self.value_plus[u] | self.value_minus[u]),
                &self.value_protected[u],
                &self.value_counters[u * BITS..(u + 1) * BITS],
                active,
            );
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn key_mask() -> Entity {
        Entity::from_fn(|p| p < 2048)
    }

    fn value_mask() -> Entity {
        Entity::domain_mask(4)
    }

    fn pair(rng: &mut Rng) -> (Token, Entity) {
        let key = Token::partial(rng.entity(), key_mask());
        let value = rng.entity() & value_mask();
        (key, value)
    }

    fn recall(block: &MemoryBlock, key: &Token) -> Token {
        let mut s = State::from_token(key, 1);
        block.forward(key, &mut s);
        s.token()
    }

    #[test]
    fn new_block_is_the_identity() {
        let mut rng = Rng::new(31);
        let block = MemoryBlock::new(16, MemoryParams::default());
        let t = Token::full(rng.entity());
        let mut s = State::from_token(&t, 1);
        let before = s.clone();
        assert_eq!(block.forward(&t, &mut s), 0);
        assert_eq!(s, before);
    }

    #[test]
    fn stores_and_recalls_with_noise() {
        let mut rng = Rng::new(32);
        let mut block = MemoryBlock::new(64, MemoryParams::default());
        let pairs: Vec<_> = (0..50).map(|_| pair(&mut rng)).collect();
        for (key, value) in &pairs {
            let s = State::from_token(key, 1);
            let up = block.learn(key, &s, value, &value_mask());
            assert_eq!(up.recruited, 1);
        }
        for (key, value) in &pairs {
            // Exact key.
            let out = recall(&block, key);
            assert_eq!(out.bits & value_mask(), *value);
            assert_eq!(out.known & value_mask(), value_mask());
            // 20% of the key bits flipped.
            let noisy = Token::partial(key.bits.with_noise(&mut rng, 200, &key_mask()), key_mask());
            assert_eq!(recall(&block, &noisy).bits & value_mask(), *value);
        }
        // An unrelated key recalls nothing.
        let (other, _) = pair(&mut rng);
        assert!((recall(&block, &other).known & value_mask()).is_zero());
        // A second presentation recruits nothing.
        let (key, value) = &pairs[0];
        let up = block.learn(key, &State::from_token(key, 1), value, &value_mask());
        assert_eq!((up.recruited, up.fired), (0, 1));
        let c = block.census();
        assert_eq!((c.units_active, c.units_free), (50, 14));
    }

    #[test]
    fn full_block_drops_new_pairs() {
        let mut rng = Rng::new(33);
        let mut block = MemoryBlock::new(4, MemoryParams::default());
        let mut dropped = 0;
        for _ in 0..6 {
            let (key, value) = pair(&mut rng);
            let up = block.learn(&key, &State::from_token(&key, 1), &value, &value_mask());
            dropped += u32::from(up.dropped);
        }
        assert_eq!(dropped, 2);
    }

    #[test]
    fn contradicted_unit_is_blamed_until_it_stops_firing() {
        let mut rng = Rng::new(34);
        let mut params = MemoryParams::default();
        params.address.automaton.live_at = 4;
        params.address.bias_step = 2;
        let mut block = MemoryBlock::new(8, params);
        let (key, value) = pair(&mut rng);
        block.learn(&key, &State::from_token(&key, 1), &value, &value_mask());
        // The same key now asks for the opposite value.
        let opposite = !value & value_mask();
        let mut blamed = 0;
        for _ in 0..200 {
            let up = block.learn(&key, &State::from_token(&key, 1), &opposite, &value_mask());
            blamed += up.blamed;
            if recall(&block, &key).bits & value_mask() == opposite {
                break;
            }
        }
        assert!(blamed > 0);
        assert_eq!(recall(&block, &key).bits & value_mask(), opposite);
    }

    #[test]
    fn idle_units_go_dormant_and_are_revived_by_learning() {
        let mut rng = Rng::new(35);
        let mut block = MemoryBlock::new(8, MemoryParams::default());
        let (k1, v1) = pair(&mut rng);
        let (k2, v2) = pair(&mut rng);
        block.learn(&k1, &State::from_token(&k1, 1), &v1, &value_mask());
        block.learn(&k2, &State::from_token(&k2, 1), &v2, &value_mask());
        block.pin(1);
        let policy = StructurePolicy {
            idle_steps_to_dormant: 2,
            unit_budget: 8,
            min_live_units: 0,
            ..StructurePolicy::default()
        };
        // Rising error: nothing is pruned.
        for _ in 0..5 {
            assert!(block.structure_step(&policy, true).skipped);
        }
        assert_eq!(block.unit_state(0), UnitState::Active);
        block.structure_step(&policy, false);
        block.structure_step(&policy, false);
        block.structure_step(&policy, false);
        assert_eq!(block.unit_state(0), UnitState::Dormant);
        assert_eq!(block.unit_state(1), UnitState::Active, "pinned unit stays");
        // Dormant: skipped by the forward pass.
        assert!((recall(&block, &k1).known & value_mask()).is_zero());
        // Learning the same pair again wakes the unit instead of recruiting.
        let up = block.learn(&k1, &State::from_token(&k1, 1), &v1, &value_mask());
        assert_eq!((up.revived_units, up.recruited), (1, 0));
        assert_eq!(recall(&block, &k1).bits & value_mask(), v1);
    }

    #[test]
    fn batch_equals_single() {
        let mut rng = Rng::new(36);
        let mut block = MemoryBlock::new(32, MemoryParams::default());
        let pairs: Vec<_> = (0..20).map(|_| pair(&mut rng)).collect();
        for (key, value) in &pairs {
            block.learn(key, &State::from_token(key, 1), value, &value_mask());
        }
        let keys: Vec<Token> = pairs.iter().map(|(k, _)| *k).collect();
        let batch = block.forward_batch(&keys);
        for (key, out) in keys.iter().zip(&batch) {
            let mut s = State::from_token(key, 1);
            block.forward(key, &mut s);
            assert_eq!(&s, out);
        }
    }
}
