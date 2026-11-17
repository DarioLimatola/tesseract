//! Hidden learning counters and the ternary automaton they drive.
//!
//! Every connection has one small signed counter that accumulates evidence:
//! `+1` when a positive connection would have helped, `-1` when a negative
//! one would have. The counter decides the two parameter bits of the
//! connection (sign and mask) with hysteresis:
//!
//! - a dormant connection becomes live when the counter reaches `live_at`
//!   in either direction, with the sign of the evidence;
//! - a live connection changes sign only when the counter reaches `live_at`
//!   on the other side;
//! - a live connection whose counter fell below `dormant_below` is a
//!   candidate for pruning. Pruning is done by the structure policy, on a
//!   budget, never here; a pruned connection is dormant, keeps its counter
//!   and can come back;
//! - a connection whose counter reaches `protect_at` becomes protected and
//!   is never pruned.
//!
//! Counting and comparing only: no multiplication, no floating point.

use crate::entity::{BITS, Entity, WORDS};

/// Thresholds of the ternary automaton.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Automaton {
    /// Evidence needed to make a connection live, or to flip its sign.
    pub live_at: i16,
    /// A live connection below this evidence may be pruned by the policy.
    pub dormant_below: i16,
    /// Saturation of the counter.
    pub limit: i16,
    /// Evidence at which a connection becomes protected. Set above `limit`
    /// to disable automatic protection.
    pub protect_at: i16,
    /// When false the connection has no sign: only positive evidence makes
    /// it live.
    pub signed: bool,
}

impl Automaton {
    /// Evidence of a counter in favour of its current connection.
    #[inline]
    fn strength(&self, h: i16) -> i16 {
        if self.signed { h.abs() } else { h }
    }
}

/// What a growth step changed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Growth {
    /// Dormant connections that became live.
    pub revived: u32,
    /// Live connections that changed sign.
    pub flipped: u32,
    /// Connections that became protected.
    pub protected: u32,
}

impl Growth {
    /// Total number of parameter bits changed.
    pub fn changes(&self) -> u32 {
        self.revived + self.flipped
    }

    /// Adds another report to this one.
    pub fn add(&mut self, other: Growth) {
        self.revived += other.revived;
        self.flipped += other.flipped;
        self.protected += other.protected;
    }
}

/// Adds `+1` to the counters at the positions of `plus` and `-1` at the
/// positions of `minus`, saturating at `±limit`.
///
/// `counters` holds one counter per bit position (length 4096).
pub fn accumulate(counters: &mut [i16], plus: &Entity, minus: &Entity, limit: i16) {
    debug_assert_eq!(counters.len(), BITS);
    for w in 0..WORDS {
        let (pw, mw) = (plus.0[w], minus.0[w]);
        if pw | mw == 0 {
            continue;
        }
        let chunk = &mut counters[w * 64..w * 64 + 64];
        for (i, c) in chunk.iter_mut().enumerate() {
            let d = ((pw >> i) & 1) as i16 - ((mw >> i) & 1) as i16;
            *c = (*c + d).clamp(-limit, limit);
        }
    }
}

/// Reference for [`accumulate`]: one position at a time.
pub fn accumulate_reference(counters: &mut [i16], plus: &Entity, minus: &Entity, limit: i16) {
    for (p, c) in counters.iter_mut().enumerate().take(BITS) {
        let mut v = i32::from(*c);
        if plus.get(p) {
            v += 1;
        }
        if minus.get(p) {
            v -= 1;
        }
        *c = v.clamp(-i32::from(limit), i32::from(limit)) as i16;
    }
}

/// Applies the growth half of the automaton: revives dormant connections
/// with enough evidence, flips the sign of live ones with enough opposite
/// evidence, and marks long-consistent ones as protected.
///
/// Only positions selected by `touched` are examined.
pub fn grow(
    counters: &[i16],
    sign: &mut Entity,
    mask: &mut Entity,
    protected: &mut Entity,
    touched: &Entity,
    a: &Automaton,
) -> Growth {
    debug_assert_eq!(counters.len(), BITS);
    let mut report = Growth::default();
    for w in 0..WORDS {
        let mut todo = touched.0[w];
        while todo != 0 {
            let i = todo.trailing_zeros() as usize;
            todo &= todo - 1;
            let bit = 1u64 << i;
            let h = counters[w * 64 + i];
            let live = mask.0[w] & bit != 0;
            let positive = h > 0;
            let strong = a.strength(h) >= a.live_at;
            if !live {
                if strong {
                    mask.0[w] |= bit;
                    if positive || !a.signed {
                        sign.0[w] |= bit;
                    } else {
                        sign.0[w] &= !bit;
                    }
                    report.revived += 1;
                }
            } else if a.signed && strong && positive != (sign.0[w] & bit != 0) {
                sign.0[w] ^= bit;
                report.flipped += 1;
            }
            if mask.0[w] & bit != 0
                && protected.0[w] & bit == 0
                && a.strength(h) >= a.protect_at
                && (!a.signed || positive == (sign.0[w] & bit != 0))
            {
                protected.0[w] |= bit;
                report.protected += 1;
            }
        }
    }
    report
}

/// Applies the pruning half of the automaton, gently: at most `budget`
/// live, unprotected connections whose evidence is below `dormant_below`
/// become dormant, weakest first, and at least `min_live` connections stay
/// live. Counters are kept. Returns the number of connections pruned.
pub fn prune(
    counters: &[i16],
    mask: &mut Entity,
    protected: &Entity,
    a: &Automaton,
    budget: u32,
    min_live: u32,
) -> u32 {
    debug_assert_eq!(counters.len(), BITS);
    let mut live = mask.count_ones();
    if budget == 0 || live <= min_live {
        return 0;
    }
    let mut candidates: Vec<(i16, u16)> = Vec::new();
    for w in 0..WORDS {
        let mut todo = mask.0[w] & !protected.0[w];
        while todo != 0 {
            let i = todo.trailing_zeros() as usize;
            todo &= todo - 1;
            let p = w * 64 + i;
            let s = a.strength(counters[p]);
            if s < a.dormant_below {
                candidates.push((s, p as u16));
            }
        }
    }
    candidates.sort_unstable();
    let mut pruned = 0;
    for (_, p) in candidates {
        if pruned >= budget || live <= min_live {
            break;
        }
        mask.set(p as usize, false);
        live -= 1;
        pruned += 1;
    }
    pruned
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    const A: Automaton = Automaton {
        live_at: 4,
        dormant_below: 2,
        limit: 10,
        protect_at: 10,
        signed: true,
    };

    #[test]
    fn accumulate_matches_reference() {
        let mut rng = Rng::new(5);
        let mut a = vec![0i16; BITS];
        let mut b = vec![0i16; BITS];
        for _ in 0..60 {
            let plus = rng.entity();
            let minus = rng.sparse_entity(300);
            accumulate(&mut a, &plus, &minus, 20);
            accumulate_reference(&mut b, &plus, &minus, 20);
        }
        assert_eq!(a, b);
        assert!(a.iter().all(|c| c.abs() <= 20));
    }

    #[test]
    fn automaton_cycle() {
        let mut counters = vec![0i16; BITS];
        let (mut sign, mut mask, mut prot) = (Entity::ZERO, Entity::ZERO, Entity::ZERO);
        let all = Entity::ONES;

        // Not enough evidence: stays dormant.
        counters[7] = 3;
        let g = grow(&counters, &mut sign, &mut mask, &mut prot, &all, &A);
        assert_eq!(g, Growth::default());

        // Enough negative evidence: live with sign 0.
        counters[7] = -4;
        let g = grow(&counters, &mut sign, &mut mask, &mut prot, &all, &A);
        assert_eq!(g.revived, 1);
        assert!(mask.get(7) && !sign.get(7));

        // Weak opposite evidence: hysteresis keeps the sign.
        counters[7] = 3;
        grow(&counters, &mut sign, &mut mask, &mut prot, &all, &A);
        assert!(mask.get(7) && !sign.get(7));

        // Strong opposite evidence: the sign flips.
        counters[7] = 4;
        let g = grow(&counters, &mut sign, &mut mask, &mut prot, &all, &A);
        assert_eq!(g.flipped, 1);
        assert!(sign.get(7));

        // Weak evidence: pruned to dormant, counter kept, and it can return.
        counters[7] = 1;
        assert_eq!(prune(&counters, &mut mask, &prot, &A, 8, 0), 1);
        assert!(!mask.get(7));
        counters[7] = 5;
        let g = grow(&counters, &mut sign, &mut mask, &mut prot, &all, &A);
        assert_eq!(g.revived, 1);

        // Saturated evidence: protected, and then never pruned.
        counters[7] = 10;
        let g = grow(&counters, &mut sign, &mut mask, &mut prot, &all, &A);
        assert_eq!(g.protected, 1);
        counters[7] = 0;
        assert_eq!(prune(&counters, &mut mask, &prot, &A, 8, 0), 0);
        assert!(mask.get(7));
    }

    #[test]
    fn pruning_is_gentle() {
        let counters = vec![0i16; BITS];
        let prot = Entity::ZERO;
        // Budget limits the number pruned.
        let mut mask = Entity::ONES;
        assert_eq!(prune(&counters, &mut mask, &prot, &A, 16, 0), 16);
        assert_eq!(mask.count_ones(), 4096 - 16);
        // A minimum of live connections is kept.
        let mut mask = Entity::from_fn(|p| p < 10);
        assert_eq!(prune(&counters, &mut mask, &prot, &A, 100, 8), 2);
        assert_eq!(mask.count_ones(), 8);
    }

    #[test]
    fn unsigned_automaton_ignores_negative_evidence() {
        let a = Automaton { signed: false, ..A };
        let mut counters = vec![0i16; BITS];
        let (mut sign, mut mask, mut prot) = (Entity::ZERO, Entity::ZERO, Entity::ZERO);
        counters[1] = -9;
        counters[2] = 9;
        let g = grow(
            &counters,
            &mut sign,
            &mut mask,
            &mut prot,
            &Entity::ONES,
            &a,
        );
        assert_eq!(g.revived, 1);
        assert!(!mask.get(1) && mask.get(2));
        // A live connection with negative evidence is the weakest candidate.
        let mut mask = Entity::from_fn(|p| p < 3);
        assert_eq!(prune(&counters, &mut mask, &prot, &a, 1, 0), 1);
        assert!(!mask.get(1));
    }
}
