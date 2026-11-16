//! The config file of a run. Everything about a run is in this file.

use std::path::Path;

use serde::{Deserialize, Serialize};
use tesseract_core::counters::Automaton;
use tesseract_core::{
    AttentionModel, AttentionParams, Entity, MemoryModel, MemoryParams, Model, StructurePolicy,
    UnitModel, UnitParams,
};
use tesseract_tasks::majority::OUTPUT_POSITION;
use tesseract_tasks::{
    Lookup, LookupSettings, Majority, MajoritySettings, Recall, RecallSettings, Task,
};

use crate::RunError;

/// A whole run: task, model, structure policy and budget.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Name of the run; also the name of its folder.
    pub name: String,
    /// Seed of every random choice of the run.
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// The task and its settings.
    pub task: TaskConfig,
    /// The model and its learning settings.
    pub model: ModelConfig,
    /// The structure policy.
    #[serde(default)]
    pub structure: StructureConfig,
    /// How long to run and how often to measure.
    #[serde(default)]
    pub budget: BudgetConfig,
}

fn default_seed() -> u64 {
    1
}

/// Which task to run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TaskConfig {
    /// Test 1: majority of a hidden subset of inputs.
    Majority {
        /// Size of the hidden subset.
        #[serde(default = "d_subset")]
        subset: usize,
        /// Hidden random sign per member.
        #[serde(default = "d_true")]
        hidden_signs: bool,
        /// Number of held-out examples.
        #[serde(default = "d_held_majority")]
        held_out: usize,
    },
    /// Test 2: associative recall of key-value pairs.
    Recall {
        /// Number of pairs to store.
        #[serde(default = "d_pairs")]
        pairs: usize,
        /// Share of key bits flipped in the held-out queries (0 to 1).
        #[serde(default = "d_noise")]
        noise: f64,
        /// Further noise levels reported as a curve.
        #[serde(default = "d_curve")]
        curve: Vec<f64>,
    },
    /// Test 3: attention lookup over records.
    Lookup {
        /// Records per sequence.
        #[serde(default = "d_records")]
        records: usize,
        /// Number of possible values.
        #[serde(default = "d_values")]
        values: usize,
        /// Number of possible keys.
        #[serde(default = "d_keys")]
        keys: usize,
        /// Number of misleading domains.
        #[serde(default = "d_decoys")]
        decoys: usize,
        /// Domain that holds the key.
        #[serde(default = "d_key_domain")]
        key_domain: usize,
        /// Domain that holds the value.
        #[serde(default = "d_value_domain")]
        value_domain: usize,
        /// Number of held-out sequences.
        #[serde(default = "d_held_lookup")]
        held_out: usize,
    },
}

fn d_subset() -> usize {
    65
}
fn d_true() -> bool {
    true
}
fn d_held_majority() -> usize {
    10_000
}
fn d_pairs() -> usize {
    1000
}
fn d_noise() -> f64 {
    0.10
}
fn d_curve() -> Vec<f64> {
    vec![0.0, 0.1, 0.2, 0.3, 0.35, 0.4]
}
fn d_records() -> usize {
    8
}
fn d_values() -> usize {
    16
}
fn d_keys() -> usize {
    64
}
fn d_decoys() -> usize {
    3
}
fn d_key_domain() -> usize {
    2
}
fn d_value_domain() -> usize {
    5
}
fn d_held_lookup() -> usize {
    2000
}

/// Thresholds of the counters of one kind of connection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterConfig {
    /// Evidence that makes a connection live (or flips its sign).
    pub live_at: i16,
    /// A live connection below this evidence may be pruned.
    pub dormant_below: i16,
    /// Saturation of the counter.
    pub limit: i16,
    /// Evidence that protects a connection for good. Zero disables it.
    #[serde(default)]
    pub protect_at: i16,
}

impl CounterConfig {
    fn automaton(&self, signed: bool) -> Automaton {
        let limit = self.limit.clamp(1, i16::MAX - 1);
        Automaton {
            live_at: self.live_at.clamp(1, limit),
            dormant_below: self.dormant_below.min(self.live_at),
            limit,
            protect_at: if self.protect_at > 0 {
                self.protect_at
            } else {
                limit + 1
            },
            signed,
        }
    }
}

/// Which model to build.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelConfig {
    /// One ternary counting unit.
    Unit {
        /// Counters of the input connections.
        #[serde(default = "d_unit_inputs")]
        inputs: CounterConfig,
        /// Net errors in one direction that move the bias by one.
        #[serde(default = "d_bias_step")]
        bias_step: i16,
        /// Also learn from right outputs closer to the threshold than this.
        #[serde(default)]
        margin: i32,
    },
    /// One memory block.
    Memory {
        /// Number of units.
        #[serde(default = "d_units")]
        units: usize,
        /// Share of live inputs that must agree for a recruited unit to
        /// fire (0.5 to 1).
        #[serde(default = "d_vigilance")]
        vigilance: f64,
        /// Counters of the address connections.
        #[serde(default = "d_unit_inputs")]
        address: CounterConfig,
        /// Counters of the value connections.
        #[serde(default = "d_memory_value")]
        value: CounterConfig,
        /// Net errors in one direction that move a bias by one.
        #[serde(default = "d_bias_step")]
        bias_step: i16,
        /// A fired unit is blamed when it votes against more than this
        /// share of the positions in error (0 to 1).
        #[serde(default = "d_blame")]
        blame: f64,
    },
    /// One attention head.
    Attention {
        /// Counters of the match mask.
        #[serde(default = "d_match")]
        matching: CounterConfig,
        /// Counters of the value mask.
        #[serde(default = "d_head_value")]
        value: CounterConfig,
        /// Initial evidence of the match connections (they start live).
        #[serde(default = "d_match_init")]
        match_init: i16,
        /// A right selection that wins by less than this still learns.
        #[serde(default = "d_select_margin")]
        select_margin: u32,
    },
}

fn d_unit_inputs() -> CounterConfig {
    CounterConfig {
        live_at: 64,
        dormant_below: 32,
        limit: 512,
        protect_at: 0,
    }
}
fn d_memory_value() -> CounterConfig {
    CounterConfig {
        live_at: 1,
        dormant_below: 1,
        limit: 15,
        protect_at: 0,
    }
}
fn d_match() -> CounterConfig {
    CounterConfig {
        live_at: 24,
        dormant_below: 8,
        limit: 64,
        protect_at: 0,
    }
}
fn d_head_value() -> CounterConfig {
    CounterConfig {
        live_at: 12,
        dormant_below: 4,
        limit: 64,
        protect_at: 0,
    }
}
fn d_bias_step() -> i16 {
    8
}
fn d_units() -> usize {
    4096
}
fn d_vigilance() -> f64 {
    0.625
}
fn d_blame() -> f64 {
    0.25
}
fn d_match_init() -> i16 {
    24
}
fn d_select_margin() -> u32 {
    32
}

/// The structure policy of a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructureConfig {
    /// Learning steps between two structure steps. Zero turns pruning off.
    #[serde(default = "d_period")]
    pub period: u64,
    /// Connections made dormant per structure step, at most.
    #[serde(default = "d_connection_budget")]
    pub connection_budget: u32,
    /// Units made dormant per structure step, at most.
    #[serde(default = "d_one")]
    pub unit_budget: u32,
    /// Live inputs every unit keeps.
    #[serde(default = "d_one")]
    pub min_live_inputs: u32,
    /// Active units every block keeps.
    #[serde(default = "d_one")]
    pub min_live_units: u32,
    /// Structure steps without firing after which a unit becomes dormant.
    /// Zero turns unit dormancy off.
    #[serde(default)]
    pub idle_steps_to_dormant: u32,
}

fn d_period() -> u64 {
    100
}
fn d_connection_budget() -> u32 {
    64
}
fn d_one() -> u32 {
    1
}

impl Default for StructureConfig {
    fn default() -> Self {
        Self {
            period: d_period(),
            connection_budget: d_connection_budget(),
            unit_budget: 1,
            min_live_inputs: 1,
            min_live_units: 1,
            idle_steps_to_dormant: 0,
        }
    }
}

impl StructureConfig {
    /// The policy handed to the engine.
    pub fn policy(&self) -> StructurePolicy {
        StructurePolicy {
            connection_budget: self.connection_budget,
            unit_budget: self.unit_budget,
            min_live_inputs: self.min_live_inputs,
            min_live_units: self.min_live_units,
            idle_steps_to_dormant: self.idle_steps_to_dormant,
        }
    }
}

/// How long to run and how often to measure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetConfig {
    /// Learning steps (one example each).
    #[serde(default = "d_steps")]
    pub steps: u64,
    /// Steps between two evaluations on the held-out set.
    #[serde(default = "d_eval_every")]
    pub eval_every: u64,
    /// Steps between two state snapshots. Zero keeps only the last one.
    #[serde(default)]
    pub snapshot_every: u64,
}

fn d_steps() -> u64 {
    10_000
}
fn d_eval_every() -> u64 {
    1000
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            steps: d_steps(),
            eval_every: d_eval_every(),
            snapshot_every: 0,
        }
    }
}

fn per_mille(share: f64) -> u32 {
    (share.clamp(0.0, 1.0) * 1000.0).round() as u32
}

impl Config {
    /// Reads a config from TOML text, applying `key.path=value` overrides.
    pub fn from_toml(text: &str, overrides: &[String]) -> Result<Self, RunError> {
        let mut value: toml::Value = toml::from_str(text).map_err(|e| RunError::Config(e.to_string()))?;
        for o in overrides {
            apply_override(&mut value, o)?;
        }
        let config: Config = value
            .try_into()
            .map_err(|e: toml::de::Error| RunError::Config(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    /// Reads a config file.
    pub fn load(path: &Path, overrides: &[String]) -> Result<Self, RunError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| RunError::Io(format!("{}: {e}", path.display())))?;
        Self::from_toml(&text, overrides)
    }

    /// The config as TOML text.
    pub fn to_toml(&self) -> Result<String, RunError> {
        toml::to_string_pretty(self).map_err(|e| RunError::Config(e.to_string()))
    }

    fn validate(&self) -> Result<(), RunError> {
        let bad = |m: &str| Err(RunError::Config(m.to_string()));
        if self.name.is_empty()
            || !self
                .name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return bad("name must be made of letters, digits, '-' and '_'");
        }
        if self.budget.steps == 0 || self.budget.eval_every == 0 {
            return bad("budget.steps and budget.eval_every must be positive");
        }
        match (&self.task, &self.model) {
            (TaskConfig::Majority { .. }, ModelConfig::Unit { .. })
            | (TaskConfig::Recall { .. }, ModelConfig::Memory { .. })
            | (TaskConfig::Lookup { .. }, ModelConfig::Attention { .. }) => {}
            _ => {
                return bad(
                    "this draft pairs majority with unit, recall with memory, lookup with attention",
                );
            }
        }
        if let ModelConfig::Memory { units, .. } = &self.model {
            if *units == 0 || *units > 1 << 20 {
                return bad("model.units must be between 1 and 1048576");
            }
        }
        Ok(())
    }

    /// Builds the task.
    pub fn build_task(&self) -> Box<dyn Task> {
        match &self.task {
            TaskConfig::Majority {
                subset,
                hidden_signs,
                held_out,
            } => Box::new(Majority::new(MajoritySettings {
                subset: *subset,
                hidden_signs: *hidden_signs,
                held_out: *held_out,
                seed: self.seed,
            })),
            TaskConfig::Recall {
                pairs,
                noise,
                curve,
            } => Box::new(Recall::new(RecallSettings {
                pairs: *pairs,
                noise_per_mille: per_mille(*noise),
                curve_per_mille: curve.iter().map(|n| per_mille(*n)).collect(),
                seed: self.seed,
            })),
            TaskConfig::Lookup {
                records,
                values,
                keys,
                decoys,
                key_domain,
                value_domain,
                held_out,
            } => Box::new(Lookup::new(LookupSettings {
                records: *records,
                values: *values,
                keys: *keys,
                decoys: *decoys,
                key_domain: *key_domain,
                value_domain: *value_domain,
                held_out: *held_out,
                seed: self.seed,
            })),
        }
    }

    /// Builds the model.
    pub fn build_model(&self) -> Box<dyn Model> {
        match &self.model {
            ModelConfig::Unit {
                inputs,
                bias_step,
                margin,
            } => Box::new(UnitModel::new(
                UnitParams {
                    automaton: inputs.automaton(true),
                    bias_step: (*bias_step).max(1),
                    margin: *margin,
                },
                OUTPUT_POSITION,
            )),
            ModelConfig::Memory {
                units,
                vigilance,
                address,
                value,
                bias_step,
                blame,
            } => Box::new(MemoryModel::new(
                *units,
                MemoryParams {
                    address: UnitParams {
                        automaton: address.automaton(true),
                        bias_step: (*bias_step).max(1),
                        margin: 0,
                    },
                    value: value.automaton(true),
                    vigilance_per_mille: per_mille(*vigilance),
                    output_margin: 1,
                    blame_per_mille: per_mille(*blame),
                    input_mask: Entity::ONES,
                },
            )),
            ModelConfig::Attention {
                matching,
                value,
                match_init,
                select_margin,
            } => Box::new(AttentionModel::new(AttentionParams {
                matching: matching.automaton(false),
                value: value.automaton(false),
                match_init: *match_init,
                select_margin: *select_margin,
                ..AttentionParams::default()
            })),
        }
    }
}

/// Applies one `a.b.c=value` override to a TOML tree. The value is read as
/// TOML (number, boolean, array, quoted string); anything else is taken as
/// a plain string.
fn apply_override(root: &mut toml::Value, spec: &str) -> Result<(), RunError> {
    let (path, raw) = spec
        .split_once('=')
        .ok_or_else(|| RunError::Config(format!("override '{spec}' is not key=value")))?;
    let parsed: toml::Value = toml::from_str::<toml::Table>(&format!("v = {raw}"))
        .ok()
        .and_then(|mut t| t.remove("v"))
        .unwrap_or_else(|| toml::Value::String(raw.to_string()));
    let mut node = root;
    let keys: Vec<&str> = path.trim().split('.').collect();
    for (i, key) in keys.iter().enumerate() {
        let table = node
            .as_table_mut()
            .ok_or_else(|| RunError::Config(format!("override '{spec}': '{key}' is not in a table")))?;
        if i + 1 == keys.len() {
            table.insert((*key).to_string(), parsed);
            return Ok(());
        }
        node = table
            .entry((*key).to_string())
            .or_insert_with(|| toml::Value::Table(toml::Table::new()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"
name = "t"
[task]
kind = "majority"
[model]
kind = "unit"
"#;

    #[test]
    fn defaults_and_overrides() {
        let c = Config::from_toml(MINIMAL, &[]).unwrap();
        assert_eq!(c.seed, 1);
        assert_eq!(c.budget.steps, 10_000);
        let explicit = format!(
            "{MINIMAL}[model.inputs]\nlive_at = 64\ndormant_below = 32\nlimit = 512\n"
        );
        let c = Config::from_toml(
            &explicit,
            &[
                "seed=9".into(),
                "task.subset=33".into(),
                "model.inputs.live_at=100".into(),
                "name=other".into(),
            ],
        )
        .unwrap();
        assert_eq!(c.seed, 9);
        assert_eq!(c.name, "other");
        assert!(matches!(c.task, TaskConfig::Majority { subset: 33, .. }));
        match c.model {
            ModelConfig::Unit { inputs, .. } => {
                assert_eq!(inputs.live_at, 100);
                assert_eq!(inputs.limit, 512, "untouched fields keep defaults");
            }
            _ => panic!("wrong model"),
        }
    }

    #[test]
    fn roundtrip_through_toml() {
        let c = Config::from_toml(MINIMAL, &[]).unwrap();
        let again = Config::from_toml(&c.to_toml().unwrap(), &[]).unwrap();
        assert_eq!(c, again);
    }

    #[test]
    fn bad_configs_are_errors_not_panics() {
        assert!(Config::from_toml("name = 1", &[]).is_err());
        assert!(Config::from_toml(MINIMAL, &["task.unknown=1".into()]).is_err());
        assert!(Config::from_toml(MINIMAL, &["model.kind=memory".into()]).is_err());
        assert!(Config::from_toml(MINIMAL, &["name=a/b".into()]).is_err());
        assert!(Config::from_toml(MINIMAL, &["budget.steps=0".into()]).is_err());
        assert!(Config::from_toml(MINIMAL, &["novalue".into()]).is_err());
    }
}
