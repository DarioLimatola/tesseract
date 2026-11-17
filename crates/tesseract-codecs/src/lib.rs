//! Adapters between outside data and entities.
//!
//! An adapter writes its item into the part of an entity it owns (its
//! footprint) and reads it back from a token. Several adapters with
//! different footprints can fill one entity, which is how inputs of
//! different kinds meet in the engine.
//!
//! This draft has adapters for bytes and for numbers. Grids, images and
//! audio come later.

use tesseract_core::code::{category_domain, nearest_domain, thermometer, thermometer_level};
use tesseract_core::{Entity, Token, WORDS_PER_DOMAIN};

/// A two-way adapter between an item and a part of an entity.
pub trait Codec {
    /// The outside type.
    type Item;

    /// Positions of the entity this adapter writes and reads.
    fn footprint(&self) -> Entity;

    /// Writes `item` into `token`, marking its footprint as known.
    fn encode(&self, item: &Self::Item, token: &mut Token);

    /// Reads the item back. `None` when no position of the footprint is
    /// known.
    fn decode(&self, token: &Token) -> Option<Self::Item>;
}

/// Marks a footprint as unknown in a token.
pub fn erase(token: &mut Token, footprint: &Entity) {
    token.known = token.known.and_not(footprint);
    token.bits = token.bits.and_not(footprint);
}

/// One byte as a fixed random 512-bit code filling one domain.
///
/// Bytes are unordered categories: two different bytes differ in about
/// half of their bits, so a code survives heavy noise.
#[derive(Clone, Debug)]
pub struct ByteCodec {
    domain: usize,
    codes: Vec<[u64; WORDS_PER_DOMAIN]>,
}

impl ByteCodec {
    /// An adapter for `domain`. Different seeds give unrelated codes.
    pub fn new(domain: usize, seed: u64) -> Self {
        let codes = (0..256).map(|b| category_domain(seed, b)).collect();
        Self { domain, codes }
    }

    /// The domain this adapter fills.
    pub fn domain(&self) -> usize {
        self.domain
    }

    /// The code of a byte as an entity that is zero outside the domain.
    pub fn entity(&self, byte: u8) -> Entity {
        let mut e = Entity::ZERO;
        e.set_domain(self.domain, self.codes[usize::from(byte)]);
        e
    }
}

impl Codec for ByteCodec {
    type Item = u8;

    fn footprint(&self) -> Entity {
        Entity::domain_mask(self.domain)
    }

    fn encode(&self, item: &u8, token: &mut Token) {
        token
            .bits
            .set_domain(self.domain, self.codes[usize::from(*item)]);
        token.known = token.known | self.footprint();
    }

    fn decode(&self, token: &Token) -> Option<u8> {
        if (token.known & self.footprint()).is_zero() {
            return None;
        }
        nearest_domain(&self.codes, &token.bits.domain(self.domain)).map(|i| i as u8)
    }
}

/// An integer in a range as a thermometer code in one word.
///
/// The range is divided into 64 equal steps (65 levels). Close numbers get
/// close codes: the distance between two codes is the difference of their
/// levels. In the default geometry a word is exactly one dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NumberCodec {
    domain: usize,
    word: usize,
    min: i64,
    max: i64,
}

impl NumberCodec {
    /// An adapter for one word of one domain. `max` must be greater than
    /// `min`; otherwise the range is widened to one step.
    pub fn new(domain: usize, word: usize, min: i64, max: i64) -> Self {
        let max = if max > min {
            max
        } else {
            min.saturating_add(1)
        };
        Self {
            domain,
            word,
            min,
            max,
        }
    }

    /// Level (0 to 64) of a number, clamped to the range and rounded.
    pub fn level(&self, x: i64) -> u32 {
        let span = i128::from(self.max) - i128::from(self.min);
        let offset = i128::from(x.clamp(self.min, self.max)) - i128::from(self.min);
        ((offset * 64 + span / 2) / span) as u32
    }

    /// The number at the centre of a level.
    pub fn number(&self, level: u32) -> i64 {
        let span = i128::from(self.max) - i128::from(self.min);
        let level = i128::from(level.min(64));
        (i128::from(self.min) + (level * span + 32) / 64) as i64
    }
}

impl Codec for NumberCodec {
    type Item = i64;

    fn footprint(&self) -> Entity {
        Entity::word_mask(self.domain, self.word)
    }

    fn encode(&self, item: &i64, token: &mut Token) {
        token
            .bits
            .set_word(self.domain, self.word, thermometer(self.level(*item)));
        token.known = token.known | self.footprint();
    }

    fn decode(&self, token: &Token) -> Option<i64> {
        if (token.known & self.footprint()).is_zero() {
            return None;
        }
        let level = thermometer_level(token.bits.word(self.domain, self.word));
        Some(self.number(level))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tesseract_core::Rng;

    /// Noise a byte code can take, as a share of the domain in per mille.
    ///
    /// Two random codes of `m` bits differ in about `m / 2` bits, so a
    /// code decodes correctly while the flipped bits stay well below
    /// `m / 4`. Wide domains leave room for heavy noise; a domain of one
    /// word carries 256 codes with little margin, so the share has to
    /// shrink with it. This is a capacity limit of the code, not of the
    /// engine: see the geometry sweep.
    fn survivable_noise() -> u32 {
        match tesseract_core::DOMAIN_BITS {
            512.. => 300,
            256..512 => 250,
            128..256 => 150,
            _ => 80,
        }
    }

    #[test]
    fn bytes_roundtrip_and_survive_noise() {
        let codec = ByteCodec::new(3, 7);
        let mut rng = Rng::new(1);
        let noise = survivable_noise();
        for b in 0..=255u8 {
            let mut t = Token::default();
            codec.encode(&b, &mut t);
            assert_eq!(t.known, Entity::domain_mask(3));
            assert_eq!(codec.decode(&t), Some(b));
            let noisy = Token::partial(t.bits.with_noise(&mut rng, noise, &t.known), t.known);
            assert_eq!(
                codec.decode(&noisy),
                Some(b),
                "byte {b} at {noise} per mille"
            );
        }
        assert_eq!(codec.decode(&Token::default()), None);
    }

    #[test]
    fn two_adapters_share_one_entity() {
        let letters = ByteCodec::new(0, 1);
        let size = NumberCodec::new(1, 0, 0, 640);
        let mut t = Token::default();
        letters.encode(&b'c', &mut t);
        size.encode(&320, &mut t);
        assert_eq!(letters.decode(&t), Some(b'c'));
        assert_eq!(size.decode(&t), Some(320));
        erase(&mut t, &letters.footprint());
        assert_eq!(letters.decode(&t), None);
        assert_eq!(size.decode(&t), Some(320));
    }

    #[test]
    fn numbers_are_ordered_and_clamped() {
        let codec = NumberCodec::new(2, 5, -100, 100);
        assert_eq!(codec.level(-100), 0);
        assert_eq!(codec.level(100), 64);
        assert_eq!(codec.level(-5000), 0);
        assert_eq!(codec.level(5000), 64);
        let mut last = 0;
        for x in -100..=100 {
            let level = codec.level(x);
            assert!(level >= last);
            last = level;
            let mut t = Token::default();
            codec.encode(&x, &mut t);
            let back = codec.decode(&t).unwrap();
            // One step is 200 / 64, so the error is at most 2.
            assert!((back - x).abs() <= 2, "{x} -> {back}");
        }
        // Distance between codes is the difference of levels.
        let (mut a, mut b) = (Token::default(), Token::default());
        codec.encode(&-100, &mut a);
        codec.encode(&0, &mut b);
        assert_eq!(a.bits.distance(&b.bits), 32);
    }

    #[test]
    fn degenerate_range_does_not_panic() {
        let codec = NumberCodec::new(0, 0, 5, 5);
        assert_eq!(codec.level(5), 0);
        let wide = NumberCodec::new(0, 0, i64::MIN, i64::MAX);
        assert_eq!(wide.level(i64::MAX), 64);
        assert_eq!(wide.level(i64::MIN), 0);
        let _ = wide.number(64);
    }
}
