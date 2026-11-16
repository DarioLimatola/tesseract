//! The integer state of a token: one small saturating integer per bit
//! position of an entity.
//!
//! Blocks read the state as bits and write signed votes into it. A value of
//! zero means "no evidence": the position is unknown. A block that writes
//! nothing leaves the state unchanged, so a stack of blocks starts as the
//! identity.
//!
//! This is the reference layout (one `i8` per position). Bit planes are a
//! later fast path with the same results.

use crate::entity::{BITS, Entity, WORDS};

/// Saturation limit of a state value: values stay in `-31..=31` (6 bits).
pub const STATE_LIMIT: i8 = 31;

/// A reading of a state: its sign bits and which positions are known.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Token {
    /// Bit 1 where the state is positive.
    pub bits: Entity,
    /// Bit 1 where the state is not zero.
    pub known: Entity,
}

impl Token {
    /// A token whose positions are all known.
    pub fn full(bits: Entity) -> Self {
        Self {
            bits,
            known: Entity::ONES,
        }
    }

    /// A token known only at the positions of `known`. Unknown positions
    /// read as bit 0.
    pub fn partial(bits: Entity, known: Entity) -> Self {
        Self {
            bits: bits & known,
            known,
        }
    }
}

/// 4096 saturating integers.
#[derive(Clone, PartialEq, Eq)]
#[repr(C, align(64))]
pub struct State(pub [i8; BITS]);

impl Default for State {
    fn default() -> Self {
        Self::zero()
    }
}

impl std::fmt::Debug for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let t = self.token();
        write!(
            f,
            "State(known={}, ones={})",
            t.known.count_ones(),
            t.bits.count_ones()
        )
    }
}

impl State {
    /// A state with no evidence anywhere.
    pub fn zero() -> Self {
        State([0; BITS])
    }

    /// Initial state of a token: `+weight` for a known 1, `-weight` for a
    /// known 0, zero where unknown.
    pub fn from_token(token: &Token, weight: i8) -> Self {
        let w = weight.clamp(0, STATE_LIMIT);
        let mut s = Self::zero();
        for p in 0..BITS {
            if token.known.get(p) {
                s.0[p] = if token.bits.get(p) { w } else { -w };
            }
        }
        s
    }

    /// Sign bits and known positions.
    pub fn token(&self) -> Token {
        let mut bits = Entity::ZERO;
        let mut known = Entity::ZERO;
        for w in 0..WORDS {
            let mut b = 0u64;
            let mut k = 0u64;
            for i in 0..64 {
                let v = self.0[w * 64 + i];
                b |= u64::from(v > 0) << i;
                k |= u64::from(v != 0) << i;
            }
            bits.0[w] = b;
            known.0[w] = k;
        }
        Token { bits, known }
    }

    /// Adds one vote per position: `+1` where `plus` is set, `-1` where
    /// `minus` is set (a position set in both does not move). Saturates.
    pub fn add_votes(&mut self, plus: &Entity, minus: &Entity) {
        for w in 0..WORDS {
            let (pw, mw) = (plus.0[w], minus.0[w]);
            if pw | mw == 0 {
                continue;
            }
            for i in 0..64 {
                let d = ((pw >> i) & 1) as i8 - ((mw >> i) & 1) as i8;
                let v = &mut self.0[w * 64 + i];
                *v = (*v + d).clamp(-STATE_LIMIT, STATE_LIMIT);
            }
        }
    }

    /// Reference for [`State::add_votes`]: one position at a time.
    pub fn add_votes_reference(&mut self, plus: &Entity, minus: &Entity) {
        for p in 0..BITS {
            let mut v = i32::from(self.0[p]);
            if plus.get(p) {
                v += 1;
            }
            if minus.get(p) {
                v -= 1;
            }
            self.0[p] = v.clamp(-i32::from(STATE_LIMIT), i32::from(STATE_LIMIT)) as i8;
        }
    }

    /// Positions where the state does not yet agree with `target` by at
    /// least `margin`, among those selected by `mask`. Returns
    /// `(want_up, want_down)`: positions whose value should rise or fall.
    ///
    /// This is the error signal of the output: Boolean per position, signed.
    pub fn error(&self, target: &Entity, mask: &Entity, margin: i8) -> (Entity, Entity) {
        let mut up = Entity::ZERO;
        let mut down = Entity::ZERO;
        for w in 0..WORDS {
            let m = mask.0[w];
            if m == 0 {
                continue;
            }
            let t = target.0[w];
            let mut u = 0u64;
            let mut d = 0u64;
            for i in 0..64 {
                let v = self.0[w * 64 + i];
                u |= u64::from(v < margin) << i;
                d |= u64::from(v > -margin) << i;
            }
            up.0[w] = u & t & m;
            down.0[w] = d & !t & m;
        }
        (up, down)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn token_roundtrip() {
        let mut rng = Rng::new(2);
        let bits = rng.entity();
        let known = rng.entity();
        let token = Token::partial(bits, known);
        let state = State::from_token(&token, 1);
        assert_eq!(state.token(), token);
    }

    #[test]
    fn votes_match_reference_and_saturate() {
        let mut rng = Rng::new(3);
        let mut a = State::zero();
        let mut b = State::zero();
        for _ in 0..100 {
            let plus = rng.entity();
            let minus = rng.sparse_entity(200);
            a.add_votes(&plus, &minus);
            b.add_votes_reference(&plus, &minus);
        }
        assert_eq!(a, b);
        assert!(a.0.iter().all(|v| v.abs() <= STATE_LIMIT));
        assert!(a.0.iter().any(|v| *v == STATE_LIMIT));
    }

    #[test]
    fn empty_votes_change_nothing() {
        let mut rng = Rng::new(4);
        let s = State::from_token(&Token::full(rng.entity()), 1);
        let mut t = s.clone();
        t.add_votes(&Entity::ZERO, &Entity::ZERO);
        assert_eq!(s, t);
    }

    #[test]
    fn error_signal() {
        let mut target = Entity::ZERO;
        target.set(0, true);
        target.set(1, true);
        let mut s = State::zero();
        s.0[0] = 1; // right
        s.0[1] = -1; // wrong, should rise
        s.0[2] = 1; // wrong, should fall
        s.0[3] = -1; // right
        s.0[4] = 0; // unknown, should fall
        let mask = Entity::from_fn(|p| p < 5);
        let (up, down) = s.error(&target, &mask, 1);
        assert_eq!(up, Entity::bit_mask(1));
        assert_eq!(down, Entity::bit_mask(2) | Entity::bit_mask(4));
        // Outside the mask nothing is asked.
        let (up, down) = s.error(&target, &Entity::ZERO, 1);
        assert!(up.is_zero() && down.is_zero());
    }
}
