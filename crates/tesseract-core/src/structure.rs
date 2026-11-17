//! Structure policy: the shared settings and reports of pruning.
//!
//! Pruning is gentle by rule: a small budget per period, nothing while the
//! error is rising, a dormant state before anything irreversible, and
//! protected parts that are never touched. In this draft the irreversible
//! step (removal) does not exist yet: the strongest action is "dormant".

/// Life state of a unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum UnitState {
    /// Never used; available to be recruited.
    #[default]
    Free,
    /// In use.
    Active,
    /// Masked: skipped by the forward pass, parameters and counters kept,
    /// can be revived by learning.
    Dormant,
}

/// Settings of the structure policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StructurePolicy {
    /// Largest number of connections made dormant per structure step.
    pub connection_budget: u32,
    /// Largest number of units made dormant per structure step.
    pub unit_budget: u32,
    /// Live input connections every unit keeps.
    pub min_live_inputs: u32,
    /// Active units every block keeps.
    pub min_live_units: u32,
    /// A unit that did not fire for this many consecutive structure steps
    /// becomes dormant. Zero disables unit dormancy.
    pub idle_steps_to_dormant: u32,
}

impl Default for StructurePolicy {
    fn default() -> Self {
        Self {
            connection_budget: 64,
            unit_budget: 1,
            min_live_inputs: 1,
            min_live_units: 1,
            idle_steps_to_dormant: 0,
        }
    }
}

/// What one structure step did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StructureReport {
    /// Connections made dormant.
    pub connections_pruned: u32,
    /// Units made dormant.
    pub units_pruned: u32,
    /// True when the step did nothing because the error was rising.
    pub skipped: bool,
}

/// Counts of the parts of a model, for the monitor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Census {
    /// Live connections of active units.
    pub connections_active: u64,
    /// Connections that are masked but keep evidence (a non-zero counter),
    /// plus every connection of a dormant unit.
    pub connections_dormant: u64,
    /// Removed connections. Always zero in this draft: removal is not
    /// implemented.
    pub connections_removed: u64,
    /// Protected connections.
    pub connections_protected: u64,
    /// Active units.
    pub units_active: u64,
    /// Dormant units.
    pub units_dormant: u64,
    /// Free units.
    pub units_free: u64,
}

impl Census {
    /// Adds another census to this one.
    pub fn add(&mut self, other: &Census) {
        self.connections_active += other.connections_active;
        self.connections_dormant += other.connections_dormant;
        self.connections_removed += other.connections_removed;
        self.connections_protected += other.connections_protected;
        self.units_active += other.units_active;
        self.units_dormant += other.units_dormant;
        self.units_free += other.units_free;
    }
}

/// Counts the connections of one parameter vector for a [`Census`].
pub fn count_connections(
    census: &mut Census,
    mask: &crate::entity::Entity,
    protected: &crate::entity::Entity,
    counters: &[i16],
    unit_active: bool,
) {
    let live = u64::from(mask.count_ones());
    let kept = counters
        .iter()
        .enumerate()
        .filter(|(p, c)| **c != 0 && !mask.get(*p))
        .count() as u64;
    if unit_active {
        census.connections_active += live;
        census.connections_dormant += kept;
    } else {
        census.connections_dormant += live + kept;
    }
    census.connections_protected += u64::from(protected.count_ones());
}
