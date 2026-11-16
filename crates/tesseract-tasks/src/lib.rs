//! Tasks: data generators, scoring and trivial baselines.
//!
//! A task produces examples and scores answers. It knows nothing about how
//! a model is built; a model knows nothing about tasks. Every task has a
//! held-out set, at least one trivial baseline and a pass threshold that is
//! written down before the first run.

pub mod lookup;
pub mod majority;
pub mod recall;

use tesseract_core::{Entity, Model, Token, nearest_candidate};

pub use lookup::{Lookup, LookupSettings};
pub use majority::{Majority, MajoritySettings};
pub use recall::{Recall, RecallSettings};

/// One example: a context of tokens (the last one is the query) and the
/// wanted output.
#[derive(Clone, Debug)]
pub struct Example {
    /// Tokens given to the model. The last one is the query.
    pub context: Vec<Token>,
    /// Wanted bits of the output.
    pub target: Entity,
    /// Index of the right answer among the task's candidates.
    pub answer: usize,
}

/// A trivial score to compare against.
#[derive(Clone, Debug, PartialEq)]
pub struct Baseline {
    /// Short name, used as a column in the metrics log.
    pub name: String,
    /// Score between 0 and 1 on the held-out set.
    pub score: f64,
}

/// A task: a training stream, a held-out set, candidates for the answer,
/// baselines and a pass threshold.
pub trait Task: Send + Sync {
    /// Short name of the task.
    fn name(&self) -> &'static str;

    /// Axes of the engine the task uses.
    fn axes(&self) -> &'static str;

    /// Next example of the training stream.
    fn next_example(&mut self) -> Example;

    /// The held-out set. Never produced by [`Task::next_example`].
    fn held_out(&self) -> &[Example];

    /// Further named evaluation sets (for curves), possibly none.
    fn extra_sets(&self) -> &[(String, Vec<Example>)] {
        &[]
    }

    /// The codes an answer is chosen from.
    fn candidates(&self) -> &[Entity];

    /// Positions of the output that carry the answer.
    fn target_mask(&self) -> Entity;

    /// Trivial scores on the held-out set.
    fn baselines(&self) -> Vec<Baseline>;

    /// Held-out score needed to pass.
    fn pass_threshold(&self) -> f64;
}

/// True when the model's answer to the example is the right candidate.
pub fn is_correct(model: &dyn Model, task: &dyn Task, example: &Example) -> bool {
    let output = model.forward(&example.context).token();
    nearest_candidate(&output, task.candidates(), &task.target_mask()) == Some(example.answer)
}

/// Fraction of examples answered correctly. Zero for an empty set.
pub fn score(model: &dyn Model, task: &dyn Task, examples: &[Example]) -> f64 {
    if examples.is_empty() {
        return 0.0;
    }
    let right = examples
        .iter()
        .filter(|e| is_correct(model, task, e))
        .count();
    right as f64 / examples.len() as f64
}
