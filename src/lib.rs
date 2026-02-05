//! # lidartree — Rust transpilation of lidaRtRee R package (tree_detection.R)
//!
//! Forest analysis with Airborne Laser Scanning (LiDAR) data.
//! Transpiled from R package lidaRtRee by Jean-Matthieu Monnet (INRAE), GPL-3.
//!
//! This crate provides:
//! - **Tree detection pipeline**: `maxima_detection`, `maxima_selection`,
//!   `segmentation`, `seg_adjust`, `tree_segmentation`, `tree_extraction`, `tree_detection`
//! - **Matching & evaluation**: `tree_matching`, `hist_detection`, `height_regression`
//!
//! Reference: Monnet, J.-M. 2011. *Using airborne laser scanning for mountain
//! forests mapping*. Ph.D. thesis, University of Grenoble, France.

pub mod raster;
pub mod tree_detection;
pub mod tree_matching;

pub use raster::Raster;
pub use tree_detection::*;
pub use tree_matching::*;
