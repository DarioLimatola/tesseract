//! Test 0 of the test plan: every bit operation against its slow reference
//! on random inputs, and the same results for any thread count.

use proptest::prelude::*;
use tesseract_core::counters::{accumulate, accumulate_reference};
use tesseract_core::unit::{match_count, match_count_reference};
use tesseract_core::{
    AttentionHead, AttentionParams, Axis, BITS, Entity, MemoryBlock, MemoryParams, Rng, State,
    Token, Unit,
};

/// A random entity with a random density of ones, so that sparse and dense
/// cases are both covered.
fn entity() -> impl Strategy<Value = Entity> {
    (any::<u64>(), 0u32..=1000).prop_map(|(seed, density)| {
        let mut rng = Rng::new(seed);
        match density {
            0..=100 => Entity::ZERO,
            900..=1000 => Entity::ONES,
            d => rng.sparse_entity(d),
        }
    })
}

fn axis() -> impl Strategy<Value = Axis> {
    prop_oneof![
        Just(Axis::Bit),
        Just(Axis::Byte),
        Just(Axis::Value),
        Just(Axis::Domain)
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn rotation_equals_reference(e in entity(), a in axis(), k in 0usize..20) {
        prop_assert_eq!(e.rotate(a, k), e.rotate_reference(a, k));
    }

    #[test]
    fn rotation_is_a_permutation_with_inverse(e in entity(), a in axis(), k in 0usize..8) {
        let r = e.rotate(a, k);
        prop_assert_eq!(r.count_ones(), e.count_ones());
        prop_assert_eq!(r.rotate(a, (8 - k) % 8), e);
    }

    #[test]
    fn rotations_along_different_axes_commute(e in entity(), k in 0usize..8, j in 0usize..8) {
        prop_assert_eq!(
            e.rotate(Axis::Bit, k).rotate(Axis::Domain, j),
            e.rotate(Axis::Domain, j).rotate(Axis::Bit, k)
        );
        prop_assert_eq!(
            e.rotate(Axis::Byte, k).rotate(Axis::Value, j),
            e.rotate(Axis::Value, j).rotate(Axis::Byte, k)
        );
    }

    #[test]
    fn agreement_equals_reference(a in entity(), b in entity(), m in entity()) {
        prop_assert_eq!(a.agreement(&b, &m), a.agreement_reference(&b, &m));
        prop_assert_eq!(a.agreement(&b, &Entity::ONES) + a.distance(&b), BITS as u32);
        let by_domain: u32 = (0..8).map(|d| a.domain_agreement(&b, d)).sum();
        prop_assert_eq!(by_domain, a.agreement(&b, &Entity::ONES));
    }

    #[test]
    fn match_count_equals_reference(x in entity(), s in entity(), m in entity()) {
        prop_assert_eq!(match_count(&x, &s, &m), match_count_reference(&x, &s, &m));
    }

    #[test]
    fn sensitivity_equals_reference(x in entity(), s in entity(), seed in any::<u64>(), bias in -3i32..4) {
        // Few live connections, so that the count is often at the threshold.
        let mut unit = Unit::empty();
        unit.sign = s;
        unit.mask = Rng::new(seed).sparse_entity(3);
        unit.live = unit.mask.count_ones();
        unit.bias = bias;
        prop_assert_eq!(unit.sensitivity(&x), unit.sensitivity_reference(&x));
    }

    #[test]
    fn votes_equal_reference(start in entity(), rounds in prop::collection::vec((entity(), entity()), 1..40)) {
        let mut fast = State::from_token(&Token::full(start), 1);
        let mut slow = fast.clone();
        for (plus, minus) in &rounds {
            fast.add_votes(plus, minus);
            slow.add_votes_reference(plus, minus);
        }
        prop_assert_eq!(fast, slow);
    }

    #[test]
    fn counters_equal_reference(rounds in prop::collection::vec((entity(), entity()), 1..40), limit in 1i16..12) {
        let mut fast = vec![0i16; BITS];
        let mut slow = vec![0i16; BITS];
        for (plus, minus) in &rounds {
            accumulate(&mut fast, plus, minus, limit);
            accumulate_reference(&mut slow, plus, minus, limit);
        }
        prop_assert_eq!(fast, slow);
    }

    #[test]
    fn state_token_roundtrip(bits in entity(), known in entity()) {
        let token = Token::partial(bits, known);
        prop_assert_eq!(State::from_token(&token, 1).token(), token);
    }

    #[test]
    fn attention_score_equals_reference(q in entity(), qk in entity(), c in entity(), ck in entity(), allowed in entity()) {
        let head = AttentionHead::new(AttentionParams { match_allowed: allowed, ..AttentionParams::default() });
        let (q, c) = (Token::partial(q, qk), Token::partial(c, ck));
        prop_assert_eq!(head.score(&q, &c), head.score_reference(&q, &c));
    }
}

/// A block with stored pairs and a batch of noisy queries.
fn block_and_queries() -> (MemoryBlock, Vec<Token>) {
    let mut rng = Rng::new(77);
    let key = Entity::from_fn(|p| p < 2048);
    let value = Entity::domain_mask(4);
    let mut block = MemoryBlock::new(128, MemoryParams::default());
    let mut queries = Vec::new();
    for _ in 0..100 {
        let k = Token::partial(rng.entity(), key);
        let v = rng.entity() & value;
        block.learn(&k, &State::from_token(&k, 1), &v, &value);
        queries.push(Token::partial(k.bits.with_noise(&mut rng, 150, &key), key));
    }
    (block, queries)
}

#[test]
fn same_results_for_any_thread_count() {
    let (block, queries) = block_and_queries();
    let sequential: Vec<State> = queries
        .iter()
        .map(|q| {
            let mut s = State::from_token(q, 1);
            block.forward(q, &mut s);
            s
        })
        .collect();
    for threads in [1, 2, 3, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool");
        let parallel = pool.install(|| block.forward_batch(&queries));
        assert_eq!(parallel, sequential, "{threads} threads");
    }
}

#[test]
fn same_seed_same_learning() {
    let (a, queries) = block_and_queries();
    let (b, _) = block_and_queries();
    assert_eq!(a.forward_batch(&queries), b.forward_batch(&queries));
    assert_eq!(a.census(), b.census());
}
