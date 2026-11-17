//! The entity: a four-dimensional hypercube of side 8, 8^4 = 4096 bits.
//!
//! | Level | Made of | Size |
//! |---|---|---|
//! | byte | 8 bits | 8 bits |
//! | value (one dimension) | 8 bytes | 64 bits, one processor word |
//! | domain (meta-dimension) | 8 values | 512 bits, one cache line |
//! | entity | 8 domains | 4096 bits |
//!
//! A bit position `p` in `0..4096` has four coordinates:
//! `domain = p >> 9`, `value = (p >> 6) & 7`, `byte = (p >> 3) & 7`,
//! `bit = p & 7`.
//!
//! Every operation has a slow, obvious reference implementation (suffix
//! `_reference`) and the fast paths are tested against it.

use std::fmt;
use std::ops::{BitAnd, BitOr, BitXor, Not};

/// Side of the hypercube. The same side is used at every level.
pub const SIDE: usize = 8;
/// Number of domains in an entity.
pub const DOMAINS: usize = SIDE;
/// Number of values (64-bit words) in a domain.
pub const VALUES_PER_DOMAIN: usize = SIDE;
/// Number of 64-bit words in an entity.
pub const WORDS: usize = DOMAINS * VALUES_PER_DOMAIN;
/// Number of bits in an entity.
pub const BITS: usize = WORDS * 64;
/// Number of bits in a domain.
pub const DOMAIN_BITS: usize = VALUES_PER_DOMAIN * 64;

/// One of the four axes of an entity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    /// Bits inside a byte.
    Bit,
    /// Bytes inside a value.
    Byte,
    /// Values inside a domain.
    Value,
    /// Domains inside an entity.
    Domain,
}

impl Axis {
    /// All four axes, innermost first.
    pub const ALL: [Axis; 4] = [Axis::Bit, Axis::Byte, Axis::Value, Axis::Domain];

    fn shift(self) -> usize {
        match self {
            Axis::Bit => 0,
            Axis::Byte => 3,
            Axis::Value => 6,
            Axis::Domain => 9,
        }
    }
}

/// 4096 bits, stored as 64 words, aligned to a cache line.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C, align(64))]
pub struct Entity(pub [u64; WORDS]);

impl Default for Entity {
    fn default() -> Self {
        Self::ZERO
    }
}

impl fmt::Debug for Entity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Entity(ones={}", self.count_ones())?;
        for d in 0..DOMAINS {
            write!(f, " d{}={}", d, self.domain_count(d))?;
        }
        write!(f, ")")
    }
}

impl Entity {
    /// All bits 0.
    pub const ZERO: Entity = Entity([0; WORDS]);
    /// All bits 1.
    pub const ONES: Entity = Entity([u64::MAX; WORDS]);

    /// Builds an entity from a function of the bit position.
    pub fn from_fn(mut f: impl FnMut(usize) -> bool) -> Self {
        let mut e = Self::ZERO;
        for p in 0..BITS {
            if f(p) {
                e.0[p >> 6] |= 1 << (p & 63);
            }
        }
        e
    }

    /// Reads the bit at position `p`. Positions wrap modulo 4096.
    #[inline]
    pub fn get(&self, p: usize) -> bool {
        let p = p % BITS;
        (self.0[p >> 6] >> (p & 63)) & 1 == 1
    }

    /// Writes the bit at position `p`. Positions wrap modulo 4096.
    #[inline]
    pub fn set(&mut self, p: usize, bit: bool) {
        let p = p % BITS;
        let m = 1u64 << (p & 63);
        if bit {
            self.0[p >> 6] |= m;
        } else {
            self.0[p >> 6] &= !m;
        }
    }

    /// The value (one dimension, 64 bits) at `value` in `domain`.
    #[inline]
    pub fn value(&self, domain: usize, value: usize) -> u64 {
        self.0[(domain % DOMAINS) * VALUES_PER_DOMAIN + (value % VALUES_PER_DOMAIN)]
    }

    /// Sets the value (one dimension) at `value` in `domain`.
    #[inline]
    pub fn set_value(&mut self, domain: usize, value: usize, word: u64) {
        self.0[(domain % DOMAINS) * VALUES_PER_DOMAIN + (value % VALUES_PER_DOMAIN)] = word;
    }

    /// The eight values of a domain.
    #[inline]
    pub fn domain(&self, domain: usize) -> [u64; VALUES_PER_DOMAIN] {
        let d = (domain % DOMAINS) * VALUES_PER_DOMAIN;
        let mut out = [0u64; VALUES_PER_DOMAIN];
        out.copy_from_slice(&self.0[d..d + VALUES_PER_DOMAIN]);
        out
    }

    /// Replaces the eight values of a domain.
    #[inline]
    pub fn set_domain(&mut self, domain: usize, values: [u64; VALUES_PER_DOMAIN]) {
        let d = (domain % DOMAINS) * VALUES_PER_DOMAIN;
        self.0[d..d + VALUES_PER_DOMAIN].copy_from_slice(&values);
    }

    /// A mask with every bit of one domain set.
    pub fn domain_mask(domain: usize) -> Self {
        let mut e = Self::ZERO;
        e.set_domain(domain, [u64::MAX; VALUES_PER_DOMAIN]);
        e
    }

    /// A mask with every bit of one dimension set.
    pub fn dimension_mask(domain: usize, value: usize) -> Self {
        let mut e = Self::ZERO;
        e.set_value(domain, value, u64::MAX);
        e
    }

    /// A mask with a single bit set.
    pub fn bit_mask(p: usize) -> Self {
        let mut e = Self::ZERO;
        e.set(p, true);
        e
    }

    /// Number of bits set.
    #[inline]
    pub fn count_ones(&self) -> u32 {
        let mut n = 0;
        for w in 0..WORDS {
            n += self.0[w].count_ones();
        }
        n
    }

    /// Number of bits set in one domain.
    #[inline]
    pub fn domain_count(&self, domain: usize) -> u32 {
        let d = (domain % DOMAINS) * VALUES_PER_DOMAIN;
        self.0[d..d + VALUES_PER_DOMAIN]
            .iter()
            .map(|w| w.count_ones())
            .sum()
    }

    /// True when no bit is set.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&w| w == 0)
    }

    /// Bits where the two entities agree (XNOR).
    #[inline]
    pub fn xnor(&self, other: &Entity) -> Entity {
        let mut out = Entity::ZERO;
        for w in 0..WORDS {
            out.0[w] = !(self.0[w] ^ other.0[w]);
        }
        out
    }

    /// `self AND NOT other`.
    #[inline]
    pub fn and_not(&self, other: &Entity) -> Entity {
        let mut out = Entity::ZERO;
        for w in 0..WORDS {
            out.0[w] = self.0[w] & !other.0[w];
        }
        out
    }

    /// Number of differing bits.
    #[inline]
    pub fn distance(&self, other: &Entity) -> u32 {
        let mut n = 0;
        for w in 0..WORDS {
            n += (self.0[w] ^ other.0[w]).count_ones();
        }
        n
    }

    /// Number of agreeing bits among those selected by `mask`.
    #[inline]
    pub fn agreement(&self, other: &Entity, mask: &Entity) -> u32 {
        let mut n = 0;
        for w in 0..WORDS {
            n += (!(self.0[w] ^ other.0[w]) & mask.0[w]).count_ones();
        }
        n
    }

    /// Reference for [`Entity::agreement`]: one bit at a time.
    pub fn agreement_reference(&self, other: &Entity, mask: &Entity) -> u32 {
        (0..BITS)
            .filter(|&p| mask.get(p) && self.get(p) == other.get(p))
            .count() as u32
    }

    /// Number of agreeing bits inside one domain.
    #[inline]
    pub fn domain_agreement(&self, other: &Entity, domain: usize) -> u32 {
        let d = (domain % DOMAINS) * VALUES_PER_DOMAIN;
        let mut n = 0;
        for w in d..d + VALUES_PER_DOMAIN {
            n += (!(self.0[w] ^ other.0[w])).count_ones();
        }
        n
    }

    /// Rotates the entity by `k` steps along one axis: the content at
    /// coordinate `c` on that axis moves to coordinate `(c + k) mod 8`.
    ///
    /// A rotation is a permutation of bit positions; its inverse (and its
    /// transpose for a backward pass) is the rotation by `8 - k`.
    pub fn rotate(&self, axis: Axis, k: usize) -> Entity {
        let k = k % SIDE;
        if k == 0 {
            return *self;
        }
        let mut out = Entity::ZERO;
        match axis {
            Axis::Bit => {
                // Rotate every byte left by k.
                let low = 0x0101_0101_0101_0101u64.wrapping_mul((1u64 << (SIDE - k)) - 1);
                for w in 0..WORDS {
                    let x = self.0[w];
                    out.0[w] = ((x & low) << k) | ((x & !low) >> (SIDE - k));
                }
            }
            Axis::Byte => {
                for w in 0..WORDS {
                    out.0[w] = self.0[w].rotate_left(8 * k as u32);
                }
            }
            Axis::Value => {
                for d in 0..DOMAINS {
                    for v in 0..VALUES_PER_DOMAIN {
                        out.0[d * VALUES_PER_DOMAIN + (v + k) % VALUES_PER_DOMAIN] =
                            self.0[d * VALUES_PER_DOMAIN + v];
                    }
                }
            }
            Axis::Domain => {
                for d in 0..DOMAINS {
                    for v in 0..VALUES_PER_DOMAIN {
                        out.0[((d + k) % DOMAINS) * VALUES_PER_DOMAIN + v] =
                            self.0[d * VALUES_PER_DOMAIN + v];
                    }
                }
            }
        }
        out
    }

    /// Reference for [`Entity::rotate`]: moves one bit at a time.
    pub fn rotate_reference(&self, axis: Axis, k: usize) -> Entity {
        let shift = axis.shift();
        let mut out = Entity::ZERO;
        for p in 0..BITS {
            if self.get(p) {
                let c = (p >> shift) & 7;
                let moved = (c + k) % SIDE;
                let q = (p & !(7 << shift)) | (moved << shift);
                out.set(q, true);
            }
        }
        out
    }

    /// Flips each bit independently with probability `per_mille / 1000`,
    /// only at positions selected by `within`.
    pub fn with_noise(&self, rng: &mut crate::rng::Rng, per_mille: u32, within: &Entity) -> Entity {
        let flips = rng.sparse_entity(per_mille);
        *self ^ (flips & *within)
    }
}

impl BitAnd for Entity {
    type Output = Entity;
    #[inline]
    fn bitand(self, rhs: Entity) -> Entity {
        let mut out = Entity::ZERO;
        for w in 0..WORDS {
            out.0[w] = self.0[w] & rhs.0[w];
        }
        out
    }
}

impl BitOr for Entity {
    type Output = Entity;
    #[inline]
    fn bitor(self, rhs: Entity) -> Entity {
        let mut out = Entity::ZERO;
        for w in 0..WORDS {
            out.0[w] = self.0[w] | rhs.0[w];
        }
        out
    }
}

impl BitXor for Entity {
    type Output = Entity;
    #[inline]
    fn bitxor(self, rhs: Entity) -> Entity {
        let mut out = Entity::ZERO;
        for w in 0..WORDS {
            out.0[w] = self.0[w] ^ rhs.0[w];
        }
        out
    }
}

impl Not for Entity {
    type Output = Entity;
    #[inline]
    fn not(self) -> Entity {
        let mut out = Entity::ZERO;
        for w in 0..WORDS {
            out.0[w] = !self.0[w];
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn sizes() {
        assert_eq!(BITS, 4096);
        assert_eq!(std::mem::size_of::<Entity>(), 512);
        assert_eq!(std::mem::align_of::<Entity>(), 64);
    }

    #[test]
    fn get_set_roundtrip() {
        let mut e = Entity::ZERO;
        e.set(0, true);
        e.set(4095, true);
        e.set(513, true);
        assert!(e.get(0) && e.get(4095) && e.get(513));
        assert_eq!(e.count_ones(), 3);
        assert_eq!(e.domain_count(1), 1);
        e.set(513, false);
        assert_eq!(e.count_ones(), 2);
    }

    #[test]
    fn rotation_matches_reference_and_inverts() {
        let mut rng = Rng::new(11);
        for _ in 0..20 {
            let e = rng.entity();
            for axis in Axis::ALL {
                for k in 0..=8 {
                    let fast = e.rotate(axis, k);
                    assert_eq!(fast, e.rotate_reference(axis, k), "{axis:?} {k}");
                    assert_eq!(fast.rotate(axis, 8 - (k % 8)), e);
                    assert_eq!(fast.count_ones(), e.count_ones());
                }
            }
        }
    }

    #[test]
    fn masks_cover_the_entity() {
        let mut all = Entity::ZERO;
        for d in 0..DOMAINS {
            assert_eq!(Entity::domain_mask(d).count_ones(), 512);
            all = all | Entity::domain_mask(d);
        }
        assert_eq!(all, Entity::ONES);
        assert_eq!(Entity::dimension_mask(3, 2).count_ones(), 64);
    }
}
