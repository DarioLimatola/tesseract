//! The Tesseract monitor: a local desktop window with three areas.
//!
//! 1. **Setup**: choose a config file, edit its few settings, start and stop.
//! 2. **Run**: scores on training and held-out data with the baselines,
//!    flips per step, connection counts, speed.
//! 3. **Neurons**: one cell per unit; click a unit to see its connections.
//!
//! It is ordinary local software: no server, no browser. The engine runs
//! inside this process, and everything shown is also on disk in the run
//! folder, so a finished run can be opened again and replayed.

pub mod app;
pub mod chart;
pub mod theme;
pub mod view;

pub use app::MonitorApp;
