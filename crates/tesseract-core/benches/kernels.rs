//! Speed measurements of the kernels and blocks.
//!
//! Run with `cargo bench -p tesseract-core`. No optimisation is merged
//! without a before and after number from here.

use std::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use tesseract_core::counters::accumulate;
use tesseract_core::unit::match_count;
use tesseract_core::{
    AttentionHead, AttentionParams, Axis, BITS, DOMAINS, Entity, MemoryBlock, MemoryParams, Rng,
    State, Token, Unit, UnitParams,
};

/// Positions of a key and of a value, as the recall task lays them out.
fn key_mask() -> Entity {
    (0..DOMAINS / 2).fold(Entity::ZERO, |m, d| m | Entity::domain_mask(d))
}

fn value_mask() -> Entity {
    Entity::domain_mask(DOMAINS - 1)
}

fn kernels(c: &mut Criterion) {
    let mut rng = Rng::new(1);
    let (x, sign, mask) = (rng.entity(), rng.entity(), rng.entity());

    c.bench_function("entity/match_count (one entity)", |b| {
        b.iter(|| match_count(black_box(&x), black_box(&sign), black_box(&mask)))
    });
    c.bench_function("entity/match_count reference", |b| {
        b.iter(|| {
            tesseract_core::unit::match_count_reference(
                black_box(&x),
                black_box(&sign),
                black_box(&mask),
            )
        })
    });
    for axis in Axis::ALL {
        c.bench_function(&format!("entity/rotate axis {}", axis.0), |b| {
            b.iter(|| black_box(&x).rotate(axis, 1))
        });
    }

    let state = State::from_token(&Token::full(x), 1);
    c.bench_function("state/add_votes (one entity)", |b| {
        b.iter_batched_ref(
            || state.clone(),
            |s| s.add_votes(black_box(&sign), black_box(&mask)),
            BatchSize::SmallInput,
        )
    });
    c.bench_function("state/token (read sign bits)", |b| {
        b.iter(|| black_box(&state).token())
    });
    c.bench_function("counters/accumulate (one entity)", |b| {
        let mut counters = vec![0i16; BITS];
        b.iter(|| accumulate(&mut counters, black_box(&sign), black_box(&mask), 512))
    });

    let params = UnitParams::default();
    c.bench_function("unit/learn (one error-driven step)", |b| {
        let mut unit = Unit::empty();
        let mut r = Rng::new(2);
        let inputs: Vec<Entity> = (0..64).map(|_| r.entity()).collect();
        let mut i = 0;
        b.iter(|| {
            i = (i + 1) % inputs.len();
            let out = unit.output(&inputs[i]);
            unit.learn(&inputs[i], !out, &params)
        })
    });
}

fn blocks(c: &mut Criterion) {
    let mut rng = Rng::new(3);
    let key = key_mask();
    let value = value_mask();

    // Memory block: 4096 units, all committed.
    let mut block = MemoryBlock::new(4096, MemoryParams::default());
    let mut keys = Vec::new();
    for _ in 0..4096 {
        let k = Token::partial(rng.entity(), key);
        let v = rng.entity() & value;
        block.learn(&k, &State::from_token(&k, 1), &v, &value);
        keys.push(k);
    }
    c.bench_function("memory/forward, 4096 units, one token", |b| {
        let mut i = 0;
        b.iter(|| {
            i = (i + 1) % keys.len();
            let mut s = State::from_token(&keys[i], 1);
            block.forward(black_box(&keys[i]), &mut s);
            s
        })
    });
    c.bench_function("memory/match only, 4096 units, one token", |b| {
        let mut fired = Vec::new();
        let mut i = 0;
        b.iter(|| {
            i = (i + 1) % keys.len();
            block.fired(black_box(&keys[i].bits), &mut fired);
            fired.len()
        })
    });
    c.bench_function("memory/forward_batch, 4096 units, 64 tokens", |b| {
        b.iter(|| block.forward_batch(black_box(&keys[..64])))
    });
    c.bench_function("memory/learn (recruit), one pair", |b| {
        b.iter_batched_ref(
            || MemoryBlock::new(64, MemoryParams::default()),
            |m| m.learn(&keys[0], &State::from_token(&keys[0], 1), &value, &value),
            BatchSize::LargeInput,
        )
    });

    // Attention: one head, 256 candidates.
    let head = AttentionHead::new(AttentionParams::default());
    let mut context: Vec<Token> = (0..256).map(|_| Token::full(rng.entity())).collect();
    context.push(Token::partial(context[100].bits, Entity::domain_mask(0)));
    c.bench_function("attention/scores, 256 candidates, one head", |b| {
        let mut scores = Vec::new();
        b.iter(|| {
            let (q, cands) = context.split_last().unwrap();
            head.scores(black_box(q), black_box(cands), &mut scores);
            head.select(&scores)
        })
    });
    c.bench_function("attention/forward, 256 candidates, one head", |b| {
        b.iter(|| head.forward(black_box(&context)))
    });
    let target = context[100].bits & Entity::domain_mask(1);
    c.bench_function("attention/learn, 8 candidates", |b| {
        let mut head = AttentionHead::new(AttentionParams::default());
        let small: Vec<Token> = context[96..104]
            .iter()
            .copied()
            .chain(std::iter::once(context[256]))
            .collect();
        b.iter(|| head.learn(black_box(&small), &target, &Entity::domain_mask(1)))
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(30).warm_up_time(std::time::Duration::from_millis(500)).measurement_time(std::time::Duration::from_secs(2));
    targets = kernels, blocks
}
criterion_main!(benches);
