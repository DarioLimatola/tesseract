//! The ternary counting unit.
//!
//! Each input connection has a sign bit and a mask bit (present or absent).
//! The unit counts the live connections whose input agrees with the sign and
//! fires when the count reaches its threshold:
//!
//! `output = popcount((x XNOR sign) AND mask) >= threshold`
//!
//! The threshold is stored as a signed `bias` relative to half of the live
//! connections, so it keeps its meaning when connections are pruned or
//! revived: the unit fires when `matches - mismatches >= bias`.

use crate::counters::{Automaton, Growth, accumulate, grow, prune};
use crate::entity::{BITS, Entity, WORDS};

/// `popcount((x XNOR sign) AND mask)`: live connections that agree.
#[inline]
pub fn match_count(x: &Entity, sign: &Entity, mask: &Entity) -> u32 {
    let mut n = 0;
    for w in 0..WORDS {
        n += (!(x.0[w] ^ sign.0[w]) & mask.0[w]).count_ones();
    }
    n
}

/// Reference for [`match_count`]: one connection at a time.
pub fn match_count_reference(x: &Entity, sign: &Entity, mask: &Entity) -> u32 {
    (0..BITS)
        .filter(|&p| mask.get(p) && x.get(p) == sign.get(p))
        .count() as u32
}

/// Activation in signed form: `matches - mismatches - bias`. The unit fires
/// when this is not negative.
#[inline]
pub fn activation(count: u32, live: u32, bias: i32) -> i32 {
    2 * count as i32 - live as i32 - bias
}

/// The count needed to fire: the smallest `c` with `2c >= live + bias`.
#[inline]
pub fn threshold(live: u32, bias: i32) -> u32 {
    let need = live as i32 + bias;
    if need <= 0 {
        0
    } else {
        ((need + 1) / 2) as u32
    }
}

/// Learning settings of a unit's input connections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnitParams {
    /// Automaton of the input connections.
    pub automaton: Automaton,
    /// Number of net errors in one direction that moves the bias by one.
    pub bias_step: i16,
    /// The unit also learns from correct outputs whose signed activation is
    /// closer to zero than this margin. Zero means "learn from errors only".
    pub margin: i32,
}

impl Default for UnitParams {
    fn default() -> Self {
        Self {
            automaton: Automaton {
                live_at: 64,
                dormant_below: 32,
                limit: 512,
                protect_at: 512,
                signed: true,
            },
            bias_step: 8,
            margin: 0,
        }
    }
}

/// What one learning step did to a unit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UnitUpdate {
    /// True when the example produced evidence (an error or a small margin).
    pub updated: bool,
    /// Connection changes.
    pub growth: Growth,
    /// True when the bias moved.
    pub bias_moved: bool,
}

/// One ternary counting unit over an entity, with its learning counters.
#[derive(Clone, Debug)]
pub struct Unit {
    /// Sign bit of every connection.
    pub sign: Entity,
    /// Mask bit of every connection: 1 = live, 0 = dormant.
    pub mask: Entity,
    /// Protected bit of every connection: never pruned.
    pub protected: Entity,
    /// Positions this unit is allowed to connect to.
    pub allowed: Entity,
    /// Threshold as `matches - mismatches >= bias`.
    pub bias: i32,
    /// Number of live connections (kept equal to `popcount(mask)`).
    pub live: u32,
    /// Times the unit fired.
    pub usage: u64,
    /// When true the unit does not learn.
    pub frozen: bool,
    /// When true the unit is never pruned as a whole.
    pub pinned: bool,
    /// Hidden counter of every connection.
    pub counters: Vec<i16>,
    /// Hidden counter of the bias.
    pub bias_counter: i16,
}

impl Unit {
    /// A unit with no live connection. With `bias = 0` it fires on every
    /// input, with `bias = 1` on none.
    pub fn empty() -> Self {
        Self {
            sign: Entity::ZERO,
            mask: Entity::ZERO,
            protected: Entity::ZERO,
            allowed: Entity::ONES,
            bias: 0,
            live: 0,
            usage: 0,
            frozen: false,
            pinned: false,
            counters: vec![0; BITS],
            bias_counter: 0,
        }
    }

    /// Number of live connections that agree with the input.
    #[inline]
    pub fn count(&self, x: &Entity) -> u32 {
        match_count(x, &self.sign, &self.mask)
    }

    /// The count needed to fire.
    pub fn threshold(&self) -> u32 {
        threshold(self.live, self.bias)
    }

    /// Output of the unit.
    #[inline]
    pub fn output(&self, x: &Entity) -> bool {
        activation(self.count(x), self.live, self.bias) >= 0
    }

    /// Boolean sensitivity of the output to each input:
    /// `out(x) XOR out(x with that input flipped)`.
    ///
    /// Only live connections can be sensitive, and only when the count sits
    /// right at the threshold.
    pub fn sensitivity(&self, x: &Entity) -> Entity {
        let count = self.count(x);
        let a = activation(count, self.live, self.bias);
        let agree = x.xnor(&self.sign) & self.mask;
        if (0..2).contains(&a) {
            // Firing by the smallest margin: losing any agreeing input stops it.
            agree
        } else if (-2..0).contains(&a) {
            // One short: gaining any disagreeing live input fires it.
            self.mask.and_not(&agree)
        } else {
            Entity::ZERO
        }
    }

    /// Reference for [`Unit::sensitivity`]: flips each input in turn.
    pub fn sensitivity_reference(&self, x: &Entity) -> Entity {
        let out = self.output(x);
        Entity::from_fn(|p| {
            let mut y = *x;
            y.set(p, !x.get(p));
            self.output(&y) != out
        })
    }

    /// Error-driven learning from one example.
    ///
    /// When the output is wrong (or right by less than the margin), every
    /// allowed connection receives one unit of evidence: in favour of a
    /// positive connection where the input agrees with the desired output,
    /// of a negative one where it does not. The bias receives evidence from
    /// the direction of the error.
    pub fn learn(&mut self, x: &Entity, desired: bool, params: &UnitParams) -> UnitUpdate {
        let mut update = UnitUpdate::default();
        let a = activation(self.count(x), self.live, self.bias);
        let out = a >= 0;
        if out {
            self.usage += 1;
        }
        if self.frozen {
            return update;
        }
        let close = if out {
            a < params.margin
        } else {
            -a <= params.margin
        };
        if out == desired && !close {
            return update;
        }
        update.updated = true;
        let (plus, minus) = if desired {
            (*x & self.allowed, self.allowed.and_not(x))
        } else {
            (self.allowed.and_not(x), *x & self.allowed)
        };
        accumulate(&mut self.counters, &plus, &minus, params.automaton.limit);
        update.growth = grow(
            &self.counters,
            &mut self.sign,
            &mut self.mask,
            &mut self.protected,
            &self.allowed,
            &params.automaton,
        );
        self.live = self.mask.count_ones();
        if out != desired {
            self.bias_counter += if desired { -1 } else { 1 };
            if self.bias_counter.abs() >= params.bias_step {
                self.bias += i32::from(self.bias_counter.signum());
                self.bias_counter = 0;
                update.bias_moved = true;
            }
        }
        update
    }

    /// Structure step: prunes at most `budget` weak, unprotected
    /// connections to dormant, keeping at least `min_live`.
    pub fn prune(&mut self, params: &UnitParams, budget: u32, min_live: u32) -> u32 {
        let pruned = prune(
            &self.counters,
            &mut self.mask,
            &self.protected,
            &params.automaton,
            budget,
            min_live,
        );
        self.live = self.mask.count_ones();
        pruned
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn random_unit(rng: &mut Rng, live_per_mille: u32, bias: i32) -> Unit {
        let mut u = Unit::empty();
        u.sign = rng.entity();
        u.mask = rng.sparse_entity(live_per_mille);
        u.live = u.mask.count_ones();
        u.bias = bias;
        u
    }

    #[test]
    fn threshold_and_activation_agree() {
        for live in 0..40u32 {
            for bias in -45..45i32 {
                let t = threshold(live, bias);
                for count in 0..=live {
                    assert_eq!(activation(count, live, bias) >= 0, count >= t);
                }
            }
        }
    }

    #[test]
    fn count_matches_reference() {
        let mut rng = Rng::new(21);
        for _ in 0..50 {
            let (x, s, m) = (rng.entity(), rng.entity(), rng.entity());
            assert_eq!(match_count(&x, &s, &m), match_count_reference(&x, &s, &m));
        }
    }

    #[test]
    fn sensitivity_matches_reference() {
        let mut rng = Rng::new(22);
        let mut sensitive_cases = 0;
        for i in 0..60 {
            // Few live connections so that the count is often at threshold.
            let unit = random_unit(&mut rng, 4, (i % 5) - 2);
            let x = rng.entity();
            let fast = unit.sensitivity(&x);
            assert_eq!(fast, unit.sensitivity_reference(&x));
            if !fast.is_zero() {
                sensitive_cases += 1;
            }
        }
        assert!(sensitive_cases > 5, "test must exercise sensitive inputs");
    }

    #[test]
    fn empty_unit_bias_semantics() {
        let mut u = Unit::empty();
        let x = Rng::new(1).entity();
        assert!(u.output(&x));
        u.bias = 1;
        assert!(!u.output(&x));
    }

    #[test]
    fn repeated_evidence_stops_a_wrong_firing() {
        let mut rng = Rng::new(23);
        let params = UnitParams {
            automaton: Automaton {
                live_at: 3,
                dormant_below: 1,
                limit: 20,
                protect_at: 21,
                signed: true,
            },
            bias_step: 100,
            margin: 0,
        };
        let mut u = Unit::empty();
        let x = rng.entity();
        assert!(u.output(&x));
        let mut steps = 0;
        while u.output(&x) {
            u.learn(&x, false, &params);
            steps += 1;
            assert!(steps < 10);
        }
        // The same evidence makes it fire on the opposite input.
        assert!(u.output(&!x));
        assert_eq!(u.live, BITS as u32);
    }

    #[test]
    fn frozen_unit_does_not_learn() {
        let params = UnitParams::default();
        let mut u = Unit::empty();
        u.frozen = true;
        let x = Rng::new(9).entity();
        for _ in 0..1000 {
            assert!(!u.learn(&x, false, &params).updated);
        }
        assert!(u.counters.iter().all(|c| *c == 0));
    }
}
