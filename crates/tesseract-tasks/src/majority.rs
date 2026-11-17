//! Test 1: majority of a hidden subset of inputs.
//!
//! Each input is an entity of random bits. A hidden subset of positions,
//! each with a hidden sign, decides the label: 1 when most of the subset
//! agrees with its signs. All other positions are noise. The answer is one
//! bit of the output.
//!
//! Shows: one counting unit learns, and its mask finds the relevant inputs.
//! Baseline: chance.

use tesseract_core::{BITS, Entity, Rng, Token};

use crate::{Baseline, Example, Task};

/// Settings of the majority task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MajoritySettings {
    /// Size of the hidden subset. Made odd so that there are no ties.
    pub subset: usize,
    /// When true each member of the subset has a hidden random sign.
    pub hidden_signs: bool,
    /// Number of held-out examples.
    pub held_out: usize,
    /// Seed of the task.
    pub seed: u64,
}

impl Default for MajoritySettings {
    fn default() -> Self {
        Self {
            subset: 65,
            hidden_signs: true,
            held_out: 10_000,
            seed: 1,
        }
    }
}

/// The majority task.
#[derive(Clone, Debug)]
pub struct Majority {
    members: Entity,
    signs: Entity,
    subset: u32,
    stream: Rng,
    held_out: Vec<Example>,
    candidates: Vec<Entity>,
}

/// Position of the output bit. It is unknown in every input.
pub const OUTPUT_POSITION: usize = BITS - 1;

impl Majority {
    /// Builds the task: hidden subset, hidden signs and held-out set.
    pub fn new(settings: MajoritySettings) -> Self {
        let root = Rng::new(settings.seed);
        let mut pick = root.fork(1);
        let subset = (settings.subset.clamp(1, BITS - 1)) | 1;
        let mut positions: Vec<usize> = (0..BITS - 1).collect();
        pick.shuffle(&mut positions);
        let mut members = Entity::ZERO;
        for &p in &positions[..subset] {
            members.set(p, true);
        }
        let signs = if settings.hidden_signs {
            pick.entity()
        } else {
            Entity::ONES
        };
        let mut task = Self {
            members,
            signs,
            subset: subset as u32,
            stream: root.fork(2),
            held_out: Vec::new(),
            candidates: vec![Entity::ZERO, Entity::bit_mask(OUTPUT_POSITION)],
        };
        let mut held = root.fork(3);
        task.held_out = (0..settings.held_out)
            .map(|_| task.example(&mut held))
            .collect();
        task
    }

    /// The hidden subset, for inspection after a run.
    pub fn members(&self) -> &Entity {
        &self.members
    }

    /// The hidden signs, for inspection after a run.
    pub fn signs(&self) -> &Entity {
        &self.signs
    }

    fn label(&self, x: &Entity) -> bool {
        2 * x.agreement(&self.signs, &self.members) > self.subset
    }

    fn example(&self, rng: &mut Rng) -> Example {
        let known = !Entity::bit_mask(OUTPUT_POSITION);
        let token = Token::partial(rng.entity(), known);
        let label = self.label(&token.bits);
        Example {
            context: vec![token],
            target: self.candidates[usize::from(label)],
            answer: usize::from(label),
        }
    }
}

impl Task for Majority {
    fn name(&self) -> &'static str {
        "majority"
    }

    fn axes(&self) -> &'static str {
        "batch"
    }

    fn next_example(&mut self) -> Example {
        let mut rng = self.stream.clone();
        let e = self.example(&mut rng);
        self.stream = rng;
        e
    }

    fn held_out(&self) -> &[Example] {
        &self.held_out
    }

    fn candidates(&self) -> &[Entity] {
        &self.candidates
    }

    fn target_mask(&self) -> Entity {
        Entity::bit_mask(OUTPUT_POSITION)
    }

    fn baselines(&self) -> Vec<Baseline> {
        // Always answering the more common label of the held-out set.
        let ones = self.held_out.iter().filter(|e| e.answer == 1).count();
        let n = self.held_out.len().max(1);
        vec![Baseline {
            name: "chance".into(),
            score: ones.max(n - ones) as f64 / n as f64,
        }]
    }

    fn pass_threshold(&self) -> f64 {
        0.99
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_follow_the_hidden_subset() {
        let task = Majority::new(MajoritySettings {
            held_out: 500,
            ..MajoritySettings::default()
        });
        assert_eq!(task.members().count_ones(), 65);
        assert!(!task.members().get(OUTPUT_POSITION));
        let ones = task.held_out().iter().filter(|e| e.answer == 1).count();
        assert!((150..350).contains(&ones), "labels are balanced: {ones}");
        for e in task.held_out() {
            let x = e.context[0];
            assert!(!x.known.get(OUTPUT_POSITION));
            // Flipping every member flips the label.
            let flipped = x.bits ^ *task.members();
            assert_ne!(task.label(&x.bits), task.label(&flipped));
            // Flipping only noise does not.
            let noise = x.bits ^ !*task.members();
            assert_eq!(task.label(&x.bits), task.label(&noise));
        }
        let b = task.baselines();
        assert!(b[0].score >= 0.5 && b[0].score < 0.7);
    }

    #[test]
    fn same_seed_same_stream() {
        let mut a = Majority::new(MajoritySettings {
            held_out: 10,
            ..MajoritySettings::default()
        });
        let mut b = a.clone();
        for _ in 0..20 {
            let (x, y) = (a.next_example(), b.next_example());
            assert_eq!(x.context, y.context);
            assert_eq!(x.answer, y.answer);
        }
    }
}
