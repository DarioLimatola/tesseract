//! Deterministic random bits.
//!
//! The engine never uses the operating system's randomness. Every random
//! choice comes from a seed, so the same seed gives the same bits on every
//! machine, backend and thread count.

use crate::entity::{Entity, WORDS};

/// A small deterministic generator (xoshiro256** seeded through SplitMix64).
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    /// Creates a generator from a seed.
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let s = [
            splitmix(&mut sm),
            splitmix(&mut sm),
            splitmix(&mut sm),
            splitmix(&mut sm),
        ];
        Self { s }
    }

    /// Derives an independent generator for a numbered stream, without
    /// advancing this one. Used to give tasks, codes and models their own
    /// reproducible streams.
    pub fn fork(&self, stream: u64) -> Self {
        let mut sm =
            self.s[0] ^ self.s[2].rotate_left(17) ^ stream.wrapping_mul(0xD6E8_FEB8_6659_FD93);
        let s = [
            splitmix(&mut sm),
            splitmix(&mut sm),
            splitmix(&mut sm),
            splitmix(&mut sm),
        ];
        Self { s }
    }

    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform integer in `0..n`. Returns 0 when `n` is 0.
    pub fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        // Rejection sampling: unbiased, integer only.
        let zone = u64::MAX - (u64::MAX % n);
        loop {
            let v = self.next_u64();
            if v < zone {
                return v % n;
            }
        }
    }

    /// True with probability `per_mille / 1000`.
    pub fn chance_per_mille(&mut self, per_mille: u32) -> bool {
        self.below(1000) < u64::from(per_mille)
    }

    /// An entity of 4096 independent random bits.
    pub fn entity(&mut self) -> Entity {
        let mut e = Entity::ZERO;
        for w in 0..WORDS {
            e.0[w] = self.next_u64();
        }
        e
    }

    /// An entity in which each bit is 1 with probability `per_mille / 1000`.
    pub fn sparse_entity(&mut self, per_mille: u32) -> Entity {
        let mut e = Entity::ZERO;
        for w in 0..WORDS {
            let mut word = 0u64;
            for b in 0..64 {
                if self.chance_per_mille(per_mille) {
                    word |= 1 << b;
                }
            }
            e.0[w] = word;
        }
        e
    }

    /// Shuffles a slice in place (Fisher-Yates).
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            items.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_bits() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn forks_differ_and_do_not_advance_parent() {
        let parent = Rng::new(1);
        let mut f1 = parent.fork(1);
        let mut f2 = parent.fork(2);
        assert_ne!(f1.next_u64(), f2.next_u64());
        let mut again = parent.fork(1);
        let mut first = parent.fork(1);
        assert_eq!(again.next_u64(), first.next_u64());
    }

    #[test]
    fn below_stays_in_range() {
        let mut r = Rng::new(3);
        for n in [1u64, 2, 3, 7, 16, 1000] {
            for _ in 0..200 {
                assert!(r.below(n) < n);
            }
        }
        assert_eq!(r.below(0), 0);
    }
}
