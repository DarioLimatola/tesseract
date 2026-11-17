//! State snapshots: what the neuron view of the monitor reads.
//!
//! A snapshot file is the four bytes `TSNP`, a format version (u32, little
//! endian) and a postcard-encoded [`Snapshot`]. Older versions stay
//! readable: a reader checks the version before decoding.

use std::path::Path;

use serde::{Deserialize, Serialize};
use tesseract_core::{BlockView, UnitState, WORDS};

use crate::RunError;

const MAGIC: &[u8; 4] = b"TSNP";
/// Current version of the snapshot format.
pub const SNAPSHOT_VERSION: u32 = 1;

/// The state of a model at one step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    /// Learning step at which the snapshot was taken.
    pub step: u64,
    /// One entry per block of the model.
    pub blocks: Vec<BlockSnapshot>,
}

/// The state of one block.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlockSnapshot {
    /// Kind of block: `unit`, `memory` or `attention`.
    pub kind: String,
    /// Its units.
    pub units: Vec<UnitSnapshot>,
}

/// The state of one unit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UnitSnapshot {
    /// 0 = free, 1 = active, 2 = dormant.
    pub state: u8,
    /// Times the unit fired.
    pub usage: u64,
    /// Live input connections.
    pub live: u32,
    /// Bias: fires when `matches - mismatches >= bias`.
    pub bias: i32,
    /// Connections, absent for free units.
    pub detail: Option<UnitDetail>,
}

/// The connections of one unit. Each entity is 64 words.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UnitDetail {
    /// Sign bits of the inputs.
    pub input_sign: Vec<u64>,
    /// Mask bits of the inputs (1 = live).
    pub input_mask: Vec<u64>,
    /// Protected inputs.
    pub input_protected: Vec<u64>,
    /// Positions voted up.
    pub output_plus: Vec<u64>,
    /// Positions voted down.
    pub output_minus: Vec<u64>,
    /// Counters of the inputs (4096), for small models only.
    pub counters: Option<Vec<i16>>,
}

impl UnitDetail {
    /// Reads bit `p` of one of the entities of the detail.
    pub fn bit(words: &[u64], p: usize) -> bool {
        words.get(p >> 6).is_some_and(|w| (w >> (p & 63)) & 1 == 1)
    }
}

impl Snapshot {
    /// Builds a snapshot from the engine's views.
    pub fn from_views(step: u64, views: &[BlockView]) -> Self {
        let blocks = views
            .iter()
            .map(|b| BlockSnapshot {
                kind: b.kind.to_string(),
                units: b
                    .units
                    .iter()
                    .map(|u| UnitSnapshot {
                        state: match u.state {
                            UnitState::Free => 0,
                            UnitState::Active => 1,
                            UnitState::Dormant => 2,
                        },
                        usage: u.usage,
                        live: u.live,
                        bias: u.bias,
                        detail: (u.state != UnitState::Free).then(|| UnitDetail {
                            input_sign: u.input_sign.0.to_vec(),
                            input_mask: u.input_mask.0.to_vec(),
                            input_protected: u.input_protected.0.to_vec(),
                            output_plus: u.output_plus.0.to_vec(),
                            output_minus: u.output_minus.0.to_vec(),
                            counters: u.counters.clone(),
                        }),
                    })
                    .collect(),
            })
            .collect();
        Self { step, blocks }
    }

    /// Writes the snapshot to a file.
    pub fn save(&self, path: &Path) -> Result<(), RunError> {
        let mut bytes = Vec::with_capacity(1 << 16);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&SNAPSHOT_VERSION.to_le_bytes());
        let body = postcard::to_stdvec(self).map_err(|e| RunError::Format(e.to_string()))?;
        bytes.extend_from_slice(&body);
        // Write to a temporary name and rename, so that a reader watching
        // the folder never sees half a file.
        let tmp = path.with_extension("tmp");
        let io = |e: std::io::Error| RunError::Io(format!("{}: {e}", path.display()));
        std::fs::write(&tmp, bytes).map_err(io)?;
        std::fs::rename(&tmp, path).map_err(io)
    }

    /// Reads a snapshot file.
    pub fn load(path: &Path) -> Result<Self, RunError> {
        let bytes =
            std::fs::read(path).map_err(|e| RunError::Io(format!("{}: {e}", path.display())))?;
        if bytes.len() < 8 || &bytes[..4] != MAGIC {
            return Err(RunError::Format(format!(
                "{}: not a Tesseract snapshot",
                path.display()
            )));
        }
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        if version != SNAPSHOT_VERSION {
            return Err(RunError::Format(format!(
                "{}: snapshot version {version} is not supported (this build reads {SNAPSHOT_VERSION})",
                path.display()
            )));
        }
        let snapshot: Snapshot =
            postcard::from_bytes(&bytes[8..]).map_err(|e| RunError::Format(e.to_string()))?;
        for unit in snapshot.blocks.iter().flat_map(|b| &b.units) {
            if let Some(d) = &unit.detail {
                let ok = [
                    &d.input_sign,
                    &d.input_mask,
                    &d.input_protected,
                    &d.output_plus,
                    &d.output_minus,
                ]
                .iter()
                .all(|v| v.len() == WORDS);
                if !ok {
                    return Err(RunError::Format("snapshot entity of the wrong size".into()));
                }
            }
        }
        Ok(snapshot)
    }
}
