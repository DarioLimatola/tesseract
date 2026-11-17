//! The entity: a hypercube of side [`SIDE`] and [`LEVELS`] axes.
//!
//! **The geometry of the whole engine is set by the two constants below.**
//! Change them, rebuild, and every type, test and benchmark follows. Nothing
//! else in the workspace hard-codes a size.
//!
//! With the default `SIDE = 8`, `LEVELS = 4` an entity is 8⁴ = 4096 bits:
//!
//! | Level | Made of | Size | Coincides with |
//! |---|---|---|---|
//! | byte (axis 0) | 8 bits | 8 bits | a byte |
//! | value, one dimension (axis 1) | 8 bytes | 64 bits | a processor word |
//! | domain, meta-dimension (axis 2) | 8 values | 512 bits | a cache line |
//! | entity (axis 3) | 8 domains | 4096 bits | 512 bytes |
//!
//! A bit position `p` carries one digit per axis: digit `j` of `p` is
//! `(p >> (LOG_SIDE * j)) & (SIDE - 1)`. Axis 0 is the innermost, axis
//! `LEVELS - 1` the outermost (the domains).
//!
//! Constraints, checked at compile time: `SIDE` is a power of two and at
//! least 2; `LEVELS` is at least 2; an entity holds at least one 64-bit
//! word and a domain holds at least one word.
//!
//! Every operation has a slow, obvious reference implementation (suffix
//! `_reference`) and the fast paths are property-tested against it at
//! whatever geometry the crate is built with.

use std::fmt;
use std::ops::{BitAnd, BitOr, BitXor, Not};

// ---------------------------------------------------------------------
// The two numbers that set the geometry.

/// Side of the hypercube: how many of each level fit in the level above.
/// The same side is used at every level. Must be a power of two.
///
/// Fast rotations exist for 2, 4 and 8, where a digit never straddles the
/// boundary of a 64-bit word. Larger sides work and are tested, but their
/// middle axes rotate through the slow reference path.
pub const SIDE: usize = 8;

/// Number of axes inside one entity: the depth of the recursion below the
/// entity. With `SIDE = 8`, `LEVELS = 4` gives the 8⁴-bit tesseract.
pub const LEVELS: usize = 4;

// ---------------------------------------------------------------------
// Everything below is derived.

/// Bits of a digit: `log2(SIDE)`.
pub const LOG_SIDE: u32 = SIDE.trailing_zeros();
/// Number of bits in an entity: `SIDE^LEVELS`.
pub const BITS: usize = SIDE.pow(LEVELS as u32);
/// Number of 64-bit words in an entity.
pub const WORDS: usize = BITS / 64;
/// Number of domains in an entity: the outermost axis.
pub const DOMAINS: usize = SIDE;
/// Number of bits in a domain.
pub const DOMAIN_BITS: usize = BITS / SIDE;
/// Number of 64-bit words in a domain.
pub const WORDS_PER_DOMAIN: usize = DOMAIN_BITS / 64;
/// Number of bits in one dimension (the group of axes 0 and 1). Equal to
/// 64, one word, in the default geometry.
pub const DIMENSION_BITS: usize = SIDE * SIDE;

const _: () = assert!(SIDE.is_power_of_two(), "SIDE must be a power of two");
const _: () = assert!(SIDE >= 2, "SIDE must be at least 2");
const _: () = assert!(LEVELS >= 2, "LEVELS must be at least 2");
const _: () = assert!(BITS >= 64, "an entity must hold at least one 64-bit word");
const _: () = assert!(
    DOMAIN_BITS >= 64,
    "a domain must hold at least one 64-bit word"
);

/// One axis of an entity: `Axis(0)` is the innermost.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Axis(pub usize);

const fn all_axes() -> [Axis; LEVELS] {
    let mut out = [Axis(0); LEVELS];
    let mut j = 0;
    while j < LEVELS {
        out[j] = Axis(j);
        j += 1;
    }
    out
}

impl Axis {
    /// Every axis, innermost first.
    pub const ALL: [Axis; LEVELS] = all_axes();
    /// The innermost axis: bits inside a byte, in the default geometry.
    pub const INNERMOST: Axis = Axis(0);
    /// The outermost axis: the domains.
    pub const OUTERMOST: Axis = Axis(LEVELS - 1);

    /// Position of this axis's digit, in bits.
    #[inline]
    pub const fn shift(self) -> u32 {
        LOG_SIDE * self.0 as u32
    }

    /// Bit distance between two neighbouring values along this axis.
    #[inline]
    pub const fn stride(self) -> usize {
        1usize << self.shift()
    }

    /// This axis's digit of a bit position.
    #[inline]
    pub const fn digit(self, p: usize) -> usize {
        (p >> self.shift()) & (SIDE - 1)
    }
}

/// A mask of the bits of one 64-bit word whose digit along `axis` is less
/// than `m`. Only meaningful when the digit lies inside a word.
fn low_digits_mask(axis: Axis, m: usize) -> u64 {
    let stride = axis.stride();
    let period = stride * SIDE;
    let ones = m * stride;
    let mut mask: u64 = if ones >= 64 {
        u64::MAX
    } else {
        (1u64 << ones) - 1
    };
    let mut p = period;
    while p < 64 {
        mask |= mask << p;
        p *= 2;
    }
    mask
}

/// `BITS` bits, stored as `WORDS` words, aligned to a cache line.
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

    /// Reads the bit at position `p`. Positions wrap modulo [`BITS`].
    #[inline]
    pub fn get(&self, p: usize) -> bool {
        let p = p % BITS;
        (self.0[p >> 6] >> (p & 63)) & 1 == 1
    }

    /// Writes the bit at position `p`. Positions wrap modulo [`BITS`].
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

    /// The word at index `word` of `domain`. In the default geometry a word
    /// is exactly one dimension.
    #[inline]
    pub fn word(&self, domain: usize, word: usize) -> u64 {
        self.0[(domain % DOMAINS) * WORDS_PER_DOMAIN + (word % WORDS_PER_DOMAIN)]
    }

    /// Sets the word at index `word` of `domain`.
    #[inline]
    pub fn set_word(&mut self, domain: usize, word: usize, value: u64) {
        self.0[(domain % DOMAINS) * WORDS_PER_DOMAIN + (word % WORDS_PER_DOMAIN)] = value;
    }

    /// The words of a domain.
    #[inline]
    pub fn domain(&self, domain: usize) -> [u64; WORDS_PER_DOMAIN] {
        let d = (domain % DOMAINS) * WORDS_PER_DOMAIN;
        let mut out = [0u64; WORDS_PER_DOMAIN];
        out.copy_from_slice(&self.0[d..d + WORDS_PER_DOMAIN]);
        out
    }

    /// Replaces the words of a domain.
    #[inline]
    pub fn set_domain(&mut self, domain: usize, words: [u64; WORDS_PER_DOMAIN]) {
        let d = (domain % DOMAINS) * WORDS_PER_DOMAIN;
        self.0[d..d + WORDS_PER_DOMAIN].copy_from_slice(&words);
    }

    /// A mask with every bit of one domain set.
    pub fn domain_mask(domain: usize) -> Self {
        let mut e = Self::ZERO;
        e.set_domain(domain, [u64::MAX; WORDS_PER_DOMAIN]);
        e
    }

    /// A mask with every bit of one word set.
    pub fn word_mask(domain: usize, word: usize) -> Self {
        let mut e = Self::ZERO;
        e.set_word(domain, word, u64::MAX);
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
        self.0.iter().map(|w| w.count_ones()).sum()
    }

    /// Number of bits set in one domain.
    #[inline]
    pub fn domain_count(&self, domain: usize) -> u32 {
        let d = (domain % DOMAINS) * WORDS_PER_DOMAIN;
        self.0[d..d + WORDS_PER_DOMAIN]
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
        let d = (domain % DOMAINS) * WORDS_PER_DOMAIN;
        let mut n = 0;
        for w in d..d + WORDS_PER_DOMAIN {
            n += (!(self.0[w] ^ other.0[w])).count_ones();
        }
        n
    }

    /// Rotates the entity by `k` steps along one axis: the content at
    /// coordinate `c` on that axis moves to coordinate `(c + k) mod SIDE`.
    ///
    /// A rotation is a permutation of bit positions; its inverse (and its
    /// transpose for a backward pass) is the rotation by `SIDE - k`.
    ///
    /// Three cases. An axis whose digit lies entirely above bit 5 of the
    /// position moves whole words. An axis whose digit lies entirely below
    /// bit 6 is a permutation inside every word, done with two masked
    /// shifts. An axis whose digit straddles that boundary — only possible
    /// when `SIDE` is 16 or more — falls back to the reference path, which
    /// is correct but moves one bit at a time.
    pub fn rotate(&self, axis: Axis, k: usize) -> Entity {
        let k = k % SIDE;
        if k == 0 {
            return *self;
        }
        let shift = axis.shift();
        let stride = axis.stride();
        if stride >= 64 {
            // Whole-word moves.
            let step = stride / 64;
            let block = step * SIDE;
            let mut out = Entity::ZERO;
            let mut base = 0;
            while base < WORDS {
                for d in 0..SIDE {
                    let src = base + d * step;
                    let dst = base + ((d + k) % SIDE) * step;
                    out.0[dst..dst + step].copy_from_slice(&self.0[src..src + step]);
                }
                base += block;
            }
            out
        } else if shift + LOG_SIDE <= 6 {
            // Permutation inside each word: the digits below `SIDE - k`
            // move up, the rest wrap around to the bottom.
            let lo = low_digits_mask(axis, SIDE - k);
            let up = (k * stride) as u32;
            let down = ((SIDE - k) * stride) as u32;
            let mut out = Entity::ZERO;
            for w in 0..WORDS {
                let x = self.0[w];
                out.0[w] = ((x & lo) << up) | ((x & !lo) >> down);
            }
            out
        } else {
            self.rotate_reference(axis, k)
        }
    }

    /// Reference for [`Entity::rotate`]: moves one bit at a time.
    pub fn rotate_reference(&self, axis: Axis, k: usize) -> Entity {
        let shift = axis.shift();
        let field = (SIDE - 1) << shift;
        let mut out = Entity::ZERO;
        for p in 0..BITS {
            if self.get(p) {
                let moved = (axis.digit(p) + k) % SIDE;
                out.set((p & !field) | (moved << shift), true);
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
    fn geometry_is_consistent() {
        assert_eq!(BITS, SIDE.pow(LEVELS as u32));
        assert_eq!(WORDS * 64, BITS);
        assert_eq!(DOMAINS * DOMAIN_BITS, BITS);
        assert_eq!(WORDS_PER_DOMAIN * DOMAINS, WORDS);
        // Padded up to the 64-byte alignment for small entities.
        assert!(std::mem::size_of::<Entity>() >= WORDS * 8);
        assert_eq!(std::mem::size_of::<Entity>() % 64, 0);
        assert_eq!(1usize << LOG_SIDE, SIDE);
        // Digits of a position cover it exactly.
        for p in [0, 1, BITS / 3, BITS - 1] {
            let rebuilt: usize = Axis::ALL.iter().map(|a| a.digit(p) << a.shift()).sum();
            assert_eq!(rebuilt, p, "position {p}");
        }
    }

    #[test]
    fn get_set_roundtrip() {
        let mut e = Entity::ZERO;
        let last = BITS - 1;
        let middle = DOMAIN_BITS + 1;
        e.set(0, true);
        e.set(last, true);
        e.set(middle, true);
        assert!(e.get(0) && e.get(last) && e.get(middle));
        assert_eq!(e.count_ones(), 3);
        // Every bit is counted by exactly one domain.
        let by_domain: u32 = (0..DOMAINS).map(|d| e.domain_count(d)).sum();
        assert_eq!(by_domain, 3);
        assert!(e.domain_count(1) >= 1);
        e.set(middle, false);
        assert_eq!(e.count_ones(), 2);
    }

    #[test]
    fn rotation_matches_reference_and_inverts() {
        let mut rng = Rng::new(11);
        for _ in 0..20 {
            let e = rng.entity();
            for axis in Axis::ALL {
                for k in 0..=SIDE {
                    let fast = e.rotate(axis, k);
                    assert_eq!(fast, e.rotate_reference(axis, k), "{axis:?} {k}");
                    assert_eq!(fast.rotate(axis, SIDE - (k % SIDE)), e);
                    assert_eq!(fast.count_ones(), e.count_ones());
                }
            }
        }
    }

    #[test]
    fn rotation_moves_the_right_digit() {
        for axis in Axis::ALL {
            // A single bit at the origin moves one step along the axis only.
            let moved = Entity::bit_mask(0).rotate(axis, 1);
            assert_eq!(moved, Entity::bit_mask(axis.stride()), "{axis:?}");
        }
    }

    #[test]
    fn masks_cover_the_entity() {
        let mut all = Entity::ZERO;
        for d in 0..DOMAINS {
            assert_eq!(Entity::domain_mask(d).count_ones(), DOMAIN_BITS as u32);
            all = all | Entity::domain_mask(d);
        }
        assert_eq!(all, Entity::ONES);
        assert_eq!(Entity::word_mask(DOMAINS - 1, 0).count_ones(), 64);
    }

    #[test]
    fn low_digits_mask_counts_digits() {
        for axis in Axis::ALL {
            if axis.shift() + LOG_SIDE > 6 {
                continue;
            }
            for m in 0..=SIDE {
                let mask = low_digits_mask(axis, m);
                let expected = (0..64).filter(|p| axis.digit(*p) < m).count() as u32;
                assert_eq!(mask.count_ones(), expected, "{axis:?} m={m}");
            }
        }
    }
}
