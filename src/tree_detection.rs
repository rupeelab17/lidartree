//! Core tree detection pipeline from lidaRtRee (R).
//!
//! Pipeline: dem_filtering → maxima_detection → maxima_selection →
//!           segmentation (watershed) → seg_adjust → tree_extraction
//!
//! The top-level function [`tree_detection`] orchestrates the full pipeline via
//! [`tree_segmentation`] + [`tree_extraction`]. R equivalents: `tree_segmentation()`,
//! `tree_extraction()`, `tree_detection()`; parameters align with R (`dmin`, `dprop`,
//! `hmin`, `sigma`, `nl_filter`/`nl_size`, `prop`/`min.value` in seg_adjust).

use crate::raster::Raster;
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;

// ---------------------------------------------------------------------------
// Public result types
// ---------------------------------------------------------------------------

/// A single detected tree.
///
/// Contains apex position, height, dominance, surface, volume, and optional crown WKT.
/// Additional R metrics (e.g. `std_tree_metrics`, `raster_zonal_stats`) are not yet
/// implemented; extend this struct or add a separate metrics type if needed.
#[derive(Debug, Clone)]
pub struct DetectedTree {
    /// Unique tree id (1-based, matching segment id).
    pub id: u32,
    /// X coordinate of the apex (map units).
    pub x: f64,
    /// Y coordinate of the apex (map units).
    pub y: f64,
    /// Height of the apex.
    pub h: f64,
    /// Dominance radius in map units: distance to the nearest higher pixel.
    pub dom_radius: f64,
    /// Segment (crown) surface in map units².
    pub surface: f64,
    /// Segment volume: sum of pixel heights × cell area.
    pub volume: f64,
    /// Optional 2D crown polygon as WKT string.
    pub crown_wkt: Option<String>,
}

/// Result of [`tree_segmentation`].
#[derive(Debug, Clone)]
pub struct SegmentationResult {
    /// Selected local maxima (values = dominance radius in pixels; 0 = not a maximum).
    pub local_maxima: Raster,
    /// Segment ids (0 = background / not a tree).
    pub segments_id: Raster,
    /// Non-linear (median) filtered DEM.
    pub non_linear_dem: Raster,
    /// Gaussian-smoothed DEM (used for maxima detection).
    pub smoothed_dem: Raster,
    /// Filled DEM used for extraction (= non-linear DEM with NaN→0).
    pub filled_dem: Raster,
}

/// Parameters for the full tree segmentation pipeline.
///
/// Defaults match lidaRtRee (R) where applicable: `max_width` = 11, `dmin` = 0.5,
/// `dprop` = 0, `hmin` = 5.0, `sigma` = [(0.6, 0.0)], `nl_filter` = "Median", `nl_size` = 3.
/// R passes these via `...` to `dem_filtering` (nl_filter, nl_size, sigma),
/// `maxima_selection` (dmin, dprop, hmin), and `seg_adjust` (prop → crown_prop, min.value → crown_hmin).
#[derive(Debug, Clone)]
pub struct TreeSegmentationParams {
    // -- dem_filtering --
    /// Non-linear filter type: "Median" or "None"
    pub nl_filter: String,
    /// Non-linear filter kernel size (odd integer).
    pub nl_size: usize,
    /// Gaussian sigma(s). If empty, no Gaussian smoothing. Each entry is
    /// (sigma, height_threshold): sigma is applied to pixels with height ≥ height_threshold
    /// (R: same behaviour in dem_filtering). Default: single pass sigma=0.6 for all pixels.
    pub sigma: Vec<(f64, f64)>,

    // -- maxima_detection (R: max.width) --
    /// Maximum search window half-width in pixels (R: `max.width`, default 11).
    pub max_width: usize,
    /// Add jitter to break ties between identical heights.
    pub jitter: bool,

    // -- maxima_selection --
    /// Minimum distance (m) from a maximum to the nearest higher pixel.
    pub dmin: f64,
    /// Minimum distance as proportion of height to the nearest higher pixel.
    pub dprop: f64,
    /// Minimum tree-top height (m).
    pub hmin: f64,

    // -- seg_adjust (R: prop, min.value) --
    /// Minimum crown base height as proportion of apex height (R: `prop`).
    pub crown_prop: Option<f64>,
    /// Minimum absolute crown base height (R: `min.value`).
    pub crown_hmin: Option<f64>,
}

impl Default for TreeSegmentationParams {
    fn default() -> Self {
        Self {
            nl_filter: "Median".into(),
            nl_size: 3,
            sigma: vec![(0.6, 0.0)],
            max_width: 11,
            jitter: true,
            dmin: 0.5,
            dprop: 0.0,
            hmin: 5.0,
            crown_prop: None,
            crown_hmin: None,
        }
    }
}

// ---------------------------------------------------------------------------
// dem_filtering — non-linear filtering + Gaussian smoothing
// ---------------------------------------------------------------------------

/// Median filter on a raster (square kernel of size `ksize × ksize`).
pub fn median_filter(dem: &Raster, ksize: usize) -> Raster {
    let half = (ksize / 2) as isize;
    let mut out = Raster::new(dem.nrow, dem.ncol, f64::NAN);
    out.xmin = dem.xmin;
    out.xmax = dem.xmax;
    out.ymin = dem.ymin;
    out.ymax = dem.ymax;
    out.res_x = dem.res_x;
    out.res_y = dem.res_y;

    let mut buf = Vec::with_capacity(ksize * ksize);
    for r in 0..dem.nrow {
        for c in 0..dem.ncol {
            buf.clear();
            for dr in -half..=half {
                for dc in -half..=half {
                    if let Some(v) = dem.get_opt(r as isize + dr, c as isize + dc) {
                        buf.push(v);
                    }
                }
            }
            if !buf.is_empty() {
                buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                let mid = buf.len() / 2;
                out.set(r, c, buf[mid]);
            }
        }
    }
    out
}

/// Gaussian smoothing (2D separable). `sigma` in pixels.
pub fn gaussian_smooth(dem: &Raster, sigma: f64) -> Raster {
    if sigma <= 0.0 {
        return dem.clone();
    }
    let radius = (3.0 * sigma).ceil() as isize;
    let kernel: Vec<f64> = (-radius..=radius)
        .map(|i| {
            let x = i as f64;
            (-x * x / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let ksum: f64 = kernel.iter().sum();
    let kernel: Vec<f64> = kernel.iter().map(|v| v / ksum).collect();

    // Horizontal pass
    let mut tmp = dem.clone();
    for r in 0..dem.nrow {
        for c in 0..dem.ncol {
            if dem.get(r, c).is_nan() {
                continue;
            }
            let mut sum = 0.0;
            let mut wsum = 0.0;
            for (ki, di) in (-radius..=radius).enumerate() {
                let cc = c as isize + di;
                if let Some(v) = dem.get_opt(r as isize, cc) {
                    sum += v * kernel[ki];
                    wsum += kernel[ki];
                }
            }
            if wsum > 0.0 {
                tmp.set(r, c, sum / wsum);
            }
        }
    }
    // Vertical pass
    let mut out = tmp.clone();
    for r in 0..dem.nrow {
        for c in 0..dem.ncol {
            if tmp.get(r, c).is_nan() {
                continue;
            }
            let mut sum = 0.0;
            let mut wsum = 0.0;
            for (ki, di) in (-radius..=radius).enumerate() {
                let rr = r as isize + di;
                if let Some(v) = tmp.get_opt(rr, c as isize) {
                    sum += v * kernel[ki];
                    wsum += kernel[ki];
                }
            }
            if wsum > 0.0 {
                out.set(r, c, sum / wsum);
            }
        }
    }
    out
}

/// Image pre-processing: non-linear filtering + Gaussian smoothing.
///
/// Returns (non_linear_image, smoothed_image).
pub fn dem_filtering(
    dem: &Raster,
    nl_filter: &str,
    nl_size: usize,
    sigma: &[(f64, f64)],
) -> (Raster, Raster) {
    // Non-linear filter
    let nl = match nl_filter {
        "Median" | "median" => median_filter(dem, nl_size),
        _ => dem.clone(),
    };

    // Gaussian smoothing (potentially height-dependent)
    let smoothed = if sigma.is_empty() || (sigma.len() == 1 && sigma[0].0 <= 0.0) {
        nl.clone()
    } else if sigma.len() == 1 {
        gaussian_smooth(&nl, sigma[0].0)
    } else {
        // Height-dependent sigma: apply largest sigma first, then blend
        // For simplicity, apply the sigma whose height threshold is satisfied
        // (R does an iterative approach; we approximate similarly)
        let mut result = nl.clone();
        // Sort by height threshold descending
        let mut sorted_sigma = sigma.to_vec();
        sorted_sigma.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        for &(sig, h_thresh) in &sorted_sigma {
            if sig <= 0.0 {
                continue;
            }
            let smoothed_pass = gaussian_smooth(&nl, sig);
            for r in 0..nl.nrow {
                for c in 0..nl.ncol {
                    let v = nl.get(r, c);
                    if !v.is_nan() && v >= h_thresh {
                        result.set(r, c, smoothed_pass.get(r, c));
                    }
                }
            }
        }
        result
    };

    (nl, smoothed)
}

// ---------------------------------------------------------------------------
// maxima_detection — variable window local maxima
// ---------------------------------------------------------------------------

/// Detect local maxima with variable window size.
///
/// For each pixel, finds the smallest square window in which this pixel is
/// the global maximum. The output value is the half-width of that window
/// (in pixels). Zero or NaN means it's not a local maximum.
///
/// Corresponds to R's `maxima_detection(dem, max.width)`.
pub fn maxima_detection(dem: &Raster, max_width: usize, jitter: bool) -> Raster {
    let mut input = dem.clone();

    // Add tiny jitter to break ties (as in R)
    if jitter {
        // Deterministic jitter based on position
        for r in 0..input.nrow {
            for c in 0..input.ncol {
                let v = input.get(r, c);
                if !v.is_nan() {
                    // Small deterministic noise based on coordinates
                    let noise = ((r * 7 + c * 13 + 42) % 1000) as f64 * 1e-6;
                    input.set(r, c, v + noise);
                }
            }
        }
    }

    let mut out = Raster::new(dem.nrow, dem.ncol, 0.0);
    out.xmin = dem.xmin;
    out.xmax = dem.xmax;
    out.ymin = dem.ymin;
    out.ymax = dem.ymax;
    out.res_x = dem.res_x;
    out.res_y = dem.res_y;

    let half_max = max_width as isize;

    for r in 0..dem.nrow {
        for c in 0..dem.ncol {
            let center_val = input.get(r, c);
            if center_val.is_nan() || center_val <= 0.0 {
                continue;
            }

            // Expand window until we find a higher pixel or reach max_width
            let mut dominance_radius: isize = 0;
            let mut is_max = true;

            'outer: for w in 1..=half_max {
                // Check border of the square window at distance w
                for dr in -w..=w {
                    for dc in -w..=w {
                        // Only check the border ring
                        if dr.abs() != w && dc.abs() != w {
                            continue;
                        }
                        let rr = r as isize + dr;
                        let cc = c as isize + dc;
                        if let Some(v) = input.get_opt(rr, cc) {
                            if v > center_val {
                                is_max = false;
                                break 'outer;
                            }
                        }
                    }
                }
                dominance_radius = w;
            }

            if is_max && dominance_radius > 0 {
                out.set(r, c, dominance_radius as f64);
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// maxima_selection — filter maxima by height and dominance
// ---------------------------------------------------------------------------

/// Select maxima based on minimum dominance distance and height.
///
/// A maximum is kept if:
/// - `dominance_radius_m >= dmin + dprop * height`
/// - `height >= hmin`
///
/// `dem` is used to look up the height at each maximum position.
/// `maxi` is the output of [`maxima_detection`] (values = dominance radius in pixels).
pub fn maxima_selection(
    maxi: &Raster,
    dem: &Raster,
    dmin: f64,
    dprop: f64,
    hmin: f64,
) -> Raster {
    let mut out = Raster::new(maxi.nrow, maxi.ncol, 0.0);
    out.xmin = maxi.xmin;
    out.xmax = maxi.xmax;
    out.ymin = maxi.ymin;
    out.ymax = maxi.ymax;
    out.res_x = maxi.res_x;
    out.res_y = maxi.res_y;

    for r in 0..maxi.nrow {
        for c in 0..maxi.ncol {
            let dom_px = maxi.get(r, c);
            if dom_px <= 0.0 || dom_px.is_nan() {
                continue;
            }
            let h = dem.get(r, c);
            if h.is_nan() || h < hmin {
                continue;
            }
            // Convert dominance radius from pixels to meters
            let dom_m = dom_px * maxi.res_x; // assuming square pixels
            let min_dom = dmin + dprop * h;
            if dom_m >= min_dom {
                out.set(r, c, dom_px);
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// segmentation — seed-based watershed
// ---------------------------------------------------------------------------

/// Priority queue item for watershed.
#[derive(Debug)]
struct WatershedItem {
    row: usize,
    col: usize,
    height: f64,
}

impl PartialEq for WatershedItem {
    fn eq(&self, other: &Self) -> bool {
        self.height == other.height
    }
}
impl Eq for WatershedItem {}

impl PartialOrd for WatershedItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WatershedItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: higher values first (watershed grows from peaks)
        self.height
            .partial_cmp(&other.height)
            .unwrap_or(Ordering::Equal)
    }
}

/// Seed-based watershed segmentation.
///
/// Seeds are the local maxima (non-zero pixels in `maxi`). The DEM is used as
/// the "landscape" for growing. Each seed grows to neighbouring pixels in
/// decreasing height order (marker-controlled watershed).
///
/// Returns a raster of segment IDs (0 = background, >0 = tree segment).
pub fn segmentation(maxi: &Raster, dem: &Raster) -> Raster {
    let nrow = dem.nrow;
    let ncol = dem.ncol;
    let mut segments = Raster::new(nrow, ncol, 0.0);
    segments.xmin = dem.xmin;
    segments.xmax = dem.xmax;
    segments.ymin = dem.ymin;
    segments.ymax = dem.ymax;
    segments.res_x = dem.res_x;
    segments.res_y = dem.res_y;

    let mut heap = BinaryHeap::new();
    let mut visited = vec![false; nrow * ncol];

    // Assign seed IDs and enqueue
    let mut next_id: u32 = 1;
    for r in 0..nrow {
        for c in 0..ncol {
            if maxi.get(r, c) > 0.0 && !maxi.get(r, c).is_nan() {
                segments.set(r, c, next_id as f64);
                visited[r * ncol + c] = true;
                let h = dem.get(r, c);
                if !h.is_nan() {
                    heap.push(WatershedItem {
                        row: r,
                        col: c,
                        height: h,
                    });
                }
                next_id += 1;
            }
        }
    }

    // 4-connected neighbours
    let neighbors: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    while let Some(item) = heap.pop() {
        let seg_id = segments.get(item.row, item.col) as u32;
        for &(dr, dc) in &neighbors {
            let nr = item.row as isize + dr;
            let nc = item.col as isize + dc;
            if nr < 0 || nc < 0 || nr >= nrow as isize || nc >= ncol as isize {
                continue;
            }
            let nr = nr as usize;
            let nc = nc as usize;
            let idx = nr * ncol + nc;
            if visited[idx] {
                continue;
            }
            let h = dem.get(nr, nc);
            if h.is_nan() || h <= 0.0 {
                visited[idx] = true;
                continue;
            }
            // Only grow downhill or at same level
            if h <= item.height {
                segments.set(nr, nc, seg_id as f64);
                visited[idx] = true;
                heap.push(WatershedItem {
                    row: nr,
                    col: nc,
                    height: h,
                });
            }
        }
    }

    // Second pass: flood remaining unvisited positive-height pixels
    // (handles plateaus that couldn't be reached by strictly downhill growth)
    let mut changed = true;
    while changed {
        changed = false;
        for r in 0..nrow {
            for c in 0..ncol {
                let idx = r * ncol + c;
                if visited[idx] {
                    continue;
                }
                let h = dem.get(r, c);
                if h.is_nan() || h <= 0.0 {
                    visited[idx] = true;
                    continue;
                }
                // Find a visited neighbor with the highest height and assign its segment
                let mut best_seg = 0.0f64;
                let mut best_h = f64::NEG_INFINITY;
                for &(dr, dc) in &neighbors {
                    let nr = r as isize + dr;
                    let nc_i = c as isize + dc;
                    if nr < 0 || nc_i < 0 || nr >= nrow as isize || nc_i >= ncol as isize {
                        continue;
                    }
                    let nidx = nr as usize * ncol + nc_i as usize;
                    if visited[nidx] {
                        let seg = segments.get(nr as usize, nc_i as usize);
                        let nh = dem.get(nr as usize, nc_i as usize);
                        if seg > 0.0 && nh > best_h {
                            best_h = nh;
                            best_seg = seg;
                        }
                    }
                }
                if best_seg > 0.0 {
                    segments.set(r, c, best_seg);
                    visited[idx] = true;
                    changed = true;
                }
            }
        }
    }

    segments
}

// ---------------------------------------------------------------------------
// seg_adjust — remove crown base based on height proportion
// ---------------------------------------------------------------------------

/// Adjust segments by removing pixels whose height is below a fraction of the
/// apex height (simulates crown base clipping).
///
/// - `crown_prop`: minimum height of crown base as proportion of apex height.
/// - `min_value`: minimum absolute crown base height.
///
/// Pixels below the threshold are set to segment 0 (background).
pub fn seg_adjust(
    segments: &Raster,
    dem: &Raster,
    maxi: &Raster,
    crown_prop: f64,
    min_value: f64,
) -> Raster {
    let mut out = segments.clone();

    // Build mapping: segment_id → (apex_row, apex_col, apex_height)
    let mut apex_map: HashMap<u32, (usize, usize, f64)> = HashMap::new();
    for r in 0..maxi.nrow {
        for c in 0..maxi.ncol {
            let v = maxi.get(r, c);
            if v > 0.0 && !v.is_nan() {
                let seg_id = segments.get(r, c) as u32;
                if seg_id > 0 {
                    let h = dem.get(r, c);
                    apex_map.insert(seg_id, (r, c, h));
                }
            }
        }
    }

    for r in 0..out.nrow {
        for c in 0..out.ncol {
            let seg_id = out.get(r, c) as u32;
            if seg_id == 0 {
                continue;
            }
            if let Some(&(_, _, apex_h)) = apex_map.get(&seg_id) {
                let threshold = (crown_prop * apex_h).max(min_value);
                let pixel_h = dem.get(r, c);
                if pixel_h.is_nan() || pixel_h < threshold {
                    out.set(r, c, 0.0);
                }
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// tree_segmentation — full pipeline
// ---------------------------------------------------------------------------

/// Full tree segmentation pipeline on a canopy height model.
///
/// Steps:
/// 1. `dem_filtering` (median + Gaussian)
/// 2. `maxima_detection` (variable window local maxima)
/// 3. `maxima_selection` (filter by height & dominance)
/// 4. `segmentation` (seeded watershed)
/// 5. `seg_adjust` (optional crown base clipping)
///
/// If `dtm` is provided (R: `tree_segmentation(dem, dtm = ...)`), maxima detection
/// and watershed use `dem`; selection and seg_adjust use `dem - dtm` (canopy height).
pub fn tree_segmentation(
    dem: &Raster,
    params: &TreeSegmentationParams,
    dtm: Option<&Raster>,
) -> SegmentationResult {
    // 1. Filtering (on dem)
    let (nl_dem, smoothed) = dem_filtering(
        dem,
        &params.nl_filter,
        params.nl_size,
        &params.sigma,
    );

    // Filled DEM for watershed (NaN → 0, negative → 0), from dem
    let mut filled = nl_dem.clone();
    for v in filled.data.iter_mut() {
        if v.is_nan() || *v < 0.0 {
            *v = 0.0;
        }
    }

    // When DTM is provided, selection and seg_adjust use CHM = dem - dtm
    let filled_chm: Raster = if let Some(dtm_r) = dtm {
        let chm = dem.sub_raster(dtm_r);
        let mut f = chm.clone();
        for v in f.data.iter_mut() {
            if v.is_nan() || *v < 0.0 {
                *v = 0.0;
            }
        }
        f
    } else {
        filled.clone()
    };

    // 2. Maxima detection (on smoothed dem)
    let all_maxima = maxima_detection(&smoothed, params.max_width, params.jitter);

    // 3. Maxima selection (heights from filled_chm: dem or dem-dtm)
    let selected_maxima = maxima_selection(
        &all_maxima,
        &filled_chm,
        params.dmin,
        params.dprop,
        params.hmin,
    );

    // 4. Watershed segmentation (on filled dem)
    let mut segs = segmentation(&selected_maxima, &filled);

    // 5. Segment adjustment (crown base; values from filled_chm)
    if let Some(prop) = params.crown_prop {
        let min_val = params.crown_hmin.unwrap_or(0.0);
        segs = seg_adjust(&segs, &filled_chm, &selected_maxima, prop, min_val);
    }

    SegmentationResult {
        local_maxima: selected_maxima,
        segments_id: segs,
        non_linear_dem: nl_dem,
        smoothed_dem: smoothed,
        filled_dem: filled,
    }
}

// ---------------------------------------------------------------------------
// tree_extraction — extract tree attributes from segmentation result
// ---------------------------------------------------------------------------

/// Extract tree attributes from segmentation results.
///
/// For each segment: locates apex, computes surface, volume, and optionally
/// builds a WKT crown polygon. If `mask` is provided (R: `r_mask`), only trees
/// whose apex falls inside the mask (mask value non-zero, non-NaN) are returned.
pub fn tree_extraction(
    seg_result: &SegmentationResult,
    mask: Option<&Raster>,
    compute_crown: bool,
    crown_ellipse: bool,
) -> Vec<DetectedTree> {
    let segs = &seg_result.segments_id;
    let maxi = &seg_result.local_maxima;
    let dem = &seg_result.filled_dem;

    // Collect segment stats
    let mut seg_surface: HashMap<u32, f64> = HashMap::new();
    let mut seg_volume: HashMap<u32, f64> = HashMap::new();
    let mut seg_pixels: HashMap<u32, Vec<(usize, usize)>> = HashMap::new();

    let cell_area = segs.cell_area();

    for r in 0..segs.nrow {
        for c in 0..segs.ncol {
            let seg_id = segs.get(r, c) as u32;
            if seg_id == 0 {
                continue;
            }
            *seg_surface.entry(seg_id).or_insert(0.0) += cell_area;
            let h = dem.get(r, c);
            if !h.is_nan() && h > 0.0 {
                *seg_volume.entry(seg_id).or_insert(0.0) += h * cell_area;
            }
            if compute_crown {
                seg_pixels.entry(seg_id).or_default().push((r, c));
            }
        }
    }

    // Find apices
    let mut trees: Vec<DetectedTree> = Vec::new();

    for r in 0..maxi.nrow {
        for c in 0..maxi.ncol {
            let dom = maxi.get(r, c);
            if dom <= 0.0 || dom.is_nan() {
                continue;
            }
            let seg_id = segs.get(r, c) as u32;
            if seg_id == 0 {
                continue;
            }
            // R: r_mask — only include if apex is inside mask
            if let Some(m) = mask {
                if m.nrow != segs.nrow || m.ncol != segs.ncol {
                    continue;
                }
                let v = m.get(r, c);
                if v.is_nan() || v == 0.0 {
                    continue;
                }
            }
            let (x, y) = segs.rc_to_xy(r, c);
            let h = dem.get(r, c);
            let dom_m = dom * segs.res_x;

            let surface = seg_surface.get(&seg_id).copied().unwrap_or(0.0);
            let volume = seg_volume.get(&seg_id).copied().unwrap_or(0.0);

            let crown_wkt = if compute_crown {
                seg_pixels.get(&seg_id).map(|pixels| {
                    if crown_ellipse {
                        build_crown_wkt_ellipse(pixels, segs)
                    } else {
                        build_crown_wkt(pixels, segs)
                    }
                })
            } else {
                None
            };

            trees.push(DetectedTree {
                id: seg_id,
                x,
                y,
                h,
                dom_radius: dom_m,
                surface,
                volume,
                crown_wkt,
            });
        }
    }

    // Sort by id
    trees.sort_by_key(|t| t.id);
    trees
}

/// Build a simple convex-hull WKT polygon from segment pixels.
fn build_crown_wkt(pixels: &[(usize, usize)], raster: &Raster) -> String {
    if pixels.is_empty() {
        return "POLYGON EMPTY".to_string();
    }

    // Collect cell corner points for a bounding polygon
    let mut points: Vec<(f64, f64)> = Vec::new();
    for &(r, c) in pixels {
        let (cx, cy) = raster.rc_to_xy(r, c);
        let hx = raster.res_x / 2.0;
        let hy = raster.res_y / 2.0;
        points.push((cx - hx, cy - hy));
        points.push((cx + hx, cy - hy));
        points.push((cx + hx, cy + hy));
        points.push((cx - hx, cy + hy));
    }

    // Simple convex hull (gift wrapping / Jarvis march)
    let hull = convex_hull(&points);

    let mut wkt = String::from("POLYGON((");
    for (i, &(x, y)) in hull.iter().enumerate() {
        if i > 0 {
            wkt.push(',');
        }
        wkt.push_str(&format!("{:.1} {:.1}", x, y));
    }
    // Close the ring
    if let Some(&(x, y)) = hull.first() {
        wkt.push_str(&format!(",{:.1} {:.1}", x, y));
    }
    wkt.push_str("))");
    wkt
}

/// Convex hull using gift wrapping (Jarvis march).
fn convex_hull(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if points.len() < 3 {
        return points.to_vec();
    }

    // Find leftmost point
    let mut start = 0;
    for (i, p) in points.iter().enumerate() {
        if p.0 < points[start].0 || (p.0 == points[start].0 && p.1 < points[start].1) {
            start = i;
        }
    }

    let mut hull = Vec::new();
    let mut current = start;
    loop {
        hull.push(points[current]);
        let mut next = 0;
        for i in 0..points.len() {
            if i == current {
                continue;
            }
            if next == current || cross(points[current], points[next], points[i]) < 0.0 {
                next = i;
            }
        }
        current = next;
        if current == start || hull.len() > points.len() {
            break;
        }
    }
    hull
}

fn cross(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

/// Build WKT polygon approximating an ellipse from segment pixels (R: ellipses4Crown-style).
fn build_crown_wkt_ellipse(pixels: &[(usize, usize)], raster: &Raster) -> String {
    if pixels.len() < 2 {
        return "POLYGON EMPTY".to_string();
    }
    let n = pixels.len() as f64;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for &(r, c) in pixels {
        let (x, y) = raster.rc_to_xy(r, c);
        sum_x += x;
        sum_y += y;
    }
    let cx = sum_x / n;
    let cy = sum_y / n;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for &(r, c) in pixels {
        let (x, y) = raster.rc_to_xy(r, c);
        var_x += (x - cx) * (x - cx);
        var_y += (y - cy) * (y - cy);
    }
    var_x = (var_x / n).max(raster.res_x * raster.res_x);
    var_y = (var_y / n).max(raster.res_y * raster.res_y);
    let semi_x = (var_x * 2.0).sqrt();
    let semi_y = (var_y * 2.0).sqrt();
    const N_POINTS: usize = 32;
    let mut wkt = String::from("POLYGON((");
    for i in 0..N_POINTS {
        let t = (i as f64 / N_POINTS as f64) * 2.0 * std::f64::consts::PI;
        let x = cx + semi_x * t.cos();
        let y = cy + semi_y * t.sin();
        if i > 0 {
            wkt.push(',');
        }
        wkt.push_str(&format!("{:.1} {:.1}", x, y));
    }
    wkt.push_str(&format!(",{:.1} {:.1}", cx + semi_x, cy));
    wkt.push_str("))");
    wkt
}

// ---------------------------------------------------------------------------
// tree_detection — top-level function
// ---------------------------------------------------------------------------

/// Full tree detection: segmentation + extraction.
///
/// Main entry point, equivalent to R's `tree_detection()`. Optionally: `dtm` so
/// that selection and seg_adjust use canopy height (dem - dtm); `mask` (R: r_mask)
/// to restrict extraction to apices inside the mask; `crown_ellipse` for ellipse
/// crown WKT instead of convex hull.
pub fn tree_detection(
    dem: &Raster,
    params: &TreeSegmentationParams,
    compute_crown: bool,
    dtm: Option<&Raster>,
    mask: Option<&Raster>,
    crown_ellipse: bool,
) -> Vec<DetectedTree> {
    let seg_result = tree_segmentation(dem, params, dtm);
    tree_extraction(&seg_result, mask, compute_crown, crown_ellipse)
}
