//! Test 2: associative recall.
//!
//! N pairs of a random key (four domains, 2048 bits) and a random value
//! (one domain, 512 bits) are presented once each. Then every key comes
//! back with a share of its bits flipped and the value must be recalled.
//!
//! Shows: capacity of the memory block, as a curve against N and noise.
//! Reference: exact nearest-neighbour search over the stored keys.

use tesseract_core::{Entity, Rng, Token};

use crate::{Baseline, Example, Task};

/// Settings of the recall task.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecallSettings {
    /// Number of pairs to store.
    pub pairs: usize,
    /// Share of key bits flipped in the held-out queries, per mille.
    pub noise_per_mille: u32,
    /// Further noise levels (per mille) evaluated as a curve.
    pub curve_per_mille: Vec<u32>,
    /// Seed of the task.
    pub seed: u64,
}

impl Default for RecallSettings {
    fn default() -> Self {
        Self {
            pairs: 1000,
            noise_per_mille: 100,
            curve_per_mille: vec![0, 100, 200, 300, 350, 400],
            seed: 1,
        }
    }
}

/// The recall task.
#[derive(Clone, Debug)]
pub struct Recall {
    keys: Vec<Entity>,
    values: Vec<Entity>,
    cursor: usize,
    held_out: Vec<Example>,
    extra: Vec<(String, Vec<Example>)>,
    nearest_neighbour: f64,
}

/// Positions of a key: domains 0 to 3.
pub fn key_mask() -> Entity {
    Entity::domain_mask(0)
        | Entity::domain_mask(1)
        | Entity::domain_mask(2)
        | Entity::domain_mask(3)
}

/// Positions of a value: domain 4.
pub fn value_mask() -> Entity {
    Entity::domain_mask(4)
}

impl Recall {
    /// Builds the pairs and the noisy held-out queries.
    pub fn new(settings: RecallSettings) -> Self {
        let root = Rng::new(settings.seed);
        let mut gen_pairs = root.fork(1);
        let n = settings.pairs.max(1);
        let keys: Vec<Entity> = (0..n).map(|_| gen_pairs.entity() & key_mask()).collect();
        let values: Vec<Entity> = (0..n).map(|_| gen_pairs.entity() & value_mask()).collect();
        let mut task = Self {
            keys,
            values,
            cursor: 0,
            held_out: Vec::new(),
            extra: Vec::new(),
            nearest_neighbour: 0.0,
        };
        task.held_out = task.queries(&mut root.fork(2), settings.noise_per_mille);
        task.nearest_neighbour = task.nearest_neighbour_score(&task.held_out);
        for (i, &noise) in settings.curve_per_mille.iter().enumerate() {
            let set = task.queries(&mut root.fork(100 + i as u64), noise);
            task.extra.push((format!("noise_{noise:03}"), set));
        }
        task
    }

    fn queries(&self, rng: &mut Rng, noise_per_mille: u32) -> Vec<Example> {
        (0..self.keys.len())
            .map(|i| {
                let noisy = self.keys[i].with_noise(rng, noise_per_mille, &key_mask());
                Example {
                    context: vec![Token::partial(noisy, key_mask())],
                    target: self.values[i],
                    answer: i,
                }
            })
            .collect()
    }

    /// Score of exact nearest-neighbour search over the stored keys.
    pub fn nearest_neighbour_score(&self, examples: &[Example]) -> f64 {
        if examples.is_empty() {
            return 0.0;
        }
        let mask = key_mask();
        let right = examples
            .iter()
            .filter(|e| {
                let q = e.context[0].bits;
                let best = (0..self.keys.len())
                    .max_by_key(|&i| (q.agreement(&self.keys[i], &mask), std::cmp::Reverse(i)));
                best == Some(e.answer)
            })
            .count();
        right as f64 / examples.len() as f64
    }

    /// Number of pairs.
    pub fn pairs(&self) -> usize {
        self.keys.len()
    }
}

impl Task for Recall {
    fn name(&self) -> &'static str {
        "recall"
    }

    fn axes(&self) -> &'static str {
        "batch"
    }

    /// The pairs with exact keys, in order, again and again.
    fn next_example(&mut self) -> Example {
        let i = self.cursor;
        self.cursor = (self.cursor + 1) % self.keys.len();
        Example {
            context: vec![Token::partial(self.keys[i], key_mask())],
            target: self.values[i],
            answer: i,
        }
    }

    fn held_out(&self) -> &[Example] {
        &self.held_out
    }

    fn extra_sets(&self) -> &[(String, Vec<Example>)] {
        &self.extra
    }

    fn candidates(&self) -> &[Entity] {
        &self.values
    }

    fn target_mask(&self) -> Entity {
        value_mask()
    }

    fn baselines(&self) -> Vec<Baseline> {
        vec![
            Baseline {
                name: "nearest_neighbour".into(),
                score: self.nearest_neighbour,
            },
            Baseline {
                name: "chance".into(),
                score: 1.0 / self.keys.len() as f64,
            },
        ]
    }

    fn pass_threshold(&self) -> f64 {
        0.99
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queries_are_noisy_keys_and_nearest_neighbour_finds_them() {
        let task = Recall::new(RecallSettings {
            pairs: 50,
            noise_per_mille: 200,
            curve_per_mille: vec![0, 480],
            seed: 3,
        });
        assert_eq!(task.held_out().len(), 50);
        for (i, e) in task.held_out().iter().enumerate() {
            let d = e.context[0].bits.distance(&task.keys[i]);
            assert!((300..520).contains(&d), "about 20% of 2048 bits: {d}");
            assert!((e.context[0].known & value_mask()).is_zero());
        }
        assert_eq!(task.baselines()[0].score, 1.0);
        assert_eq!(task.extra_sets().len(), 2);
        assert_eq!(task.extra_sets()[0].0, "noise_000");
        // At 48% noise even exact search starts to fail.
        assert!(task.nearest_neighbour_score(&task.extra_sets()[1].1) < 1.0);
    }

    #[test]
    fn training_stream_cycles_through_exact_pairs() {
        let mut task = Recall::new(RecallSettings {
            pairs: 3,
            ..RecallSettings::default()
        });
        let answers: Vec<usize> = (0..7).map(|_| task.next_example().answer).collect();
        assert_eq!(answers, vec![0, 1, 2, 0, 1, 2, 0]);
    }
}
