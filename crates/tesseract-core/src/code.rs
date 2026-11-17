//! Codes for one dimension (64 bits) and for one domain (512 bits).
//!
//! - Ordered quantities use a thermometer code: level `v` is `v` ones,
//!   65 levels from 0 to 64.
//! - Unordered categories use a fixed random code per category.
//! - Free bits are 64 independent binary features and need no code.

use crate::entity::WORDS_PER_DOMAIN;
use crate::rng::Rng;

/// Highest thermometer level.
pub const MAX_LEVEL: u32 = 64;

/// Thermometer code of a level in `0..=64`. Larger levels saturate at 64.
#[inline]
pub fn thermometer(level: u32) -> u64 {
    if level >= MAX_LEVEL {
        u64::MAX
    } else {
        (1u64 << level) - 1
    }
}

/// Level of a thermometer code. For a noisy code this is the number of ones,
/// which is the closest level in Hamming distance up to ties.
#[inline]
pub fn thermometer_level(code: u64) -> u32 {
    code.count_ones()
}

/// "Greater than" for two exact thermometer codes: `(a AND NOT b) != 0`.
#[inline]
pub fn greater(a: u64, b: u64) -> bool {
    a & !b != 0
}

/// Distance along one dimension: `popcount(a XOR b)`.
#[inline]
pub fn distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Fixed random 64-bit code of a category. The same `(seed, id)` always
/// gives the same code.
pub fn category(seed: u64, id: u64) -> u64 {
    Rng::new(seed).fork(id).next_u64()
}

/// Fixed random 512-bit code of a category, filling a whole domain.
pub fn category_domain(seed: u64, id: u64) -> [u64; WORDS_PER_DOMAIN] {
    let mut rng = Rng::new(seed).fork(id);
    let mut out = [0u64; WORDS_PER_DOMAIN];
    for w in &mut out {
        *w = rng.next_u64();
    }
    out
}

/// Index of the code closest to `word`, or `None` when `codes` is empty.
/// Ties go to the lowest index.
pub fn nearest(codes: &[u64], word: u64) -> Option<usize> {
    let mut best: Option<(u32, usize)> = None;
    for (i, &c) in codes.iter().enumerate() {
        let d = distance(c, word);
        if best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, i));
        }
    }
    best.map(|(_, i)| i)
}

/// Index of the 512-bit code closest to `domain`, or `None` when empty.
pub fn nearest_domain(
    codes: &[[u64; WORDS_PER_DOMAIN]],
    domain: &[u64; WORDS_PER_DOMAIN],
) -> Option<usize> {
    let mut best: Option<(u32, usize)> = None;
    for (i, c) in codes.iter().enumerate() {
        let d: u32 = c.iter().zip(domain).map(|(a, b)| distance(*a, *b)).sum();
        if best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, i));
        }
    }
    best.map(|(_, i)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thermometer_levels() {
        assert_eq!(thermometer(0), 0);
        assert_eq!(thermometer(1), 1);
        assert_eq!(thermometer(64), u64::MAX);
        assert_eq!(thermometer(200), u64::MAX);
        for v in 0..=64 {
            assert_eq!(thermometer_level(thermometer(v)), v);
        }
    }

    #[test]
    fn greater_and_distance_follow_levels() {
        for a in 0..=64u32 {
            for b in 0..=64u32 {
                assert_eq!(greater(thermometer(a), thermometer(b)), a > b);
                assert_eq!(distance(thermometer(a), thermometer(b)), a.abs_diff(b));
            }
        }
    }

    #[test]
    fn categories_are_fixed_and_distinct() {
        assert_eq!(category(5, 9), category(5, 9));
        assert_ne!(category(5, 9), category(5, 10));
        assert_ne!(category(5, 9), category(6, 9));
        let codes: Vec<u64> = (0..16).map(|i| category(1, i)).collect();
        for (i, &c) in codes.iter().enumerate() {
            assert_eq!(nearest(&codes, c ^ 0b111), Some(i));
        }
        assert_eq!(nearest(&[], 0), None);
    }
}
