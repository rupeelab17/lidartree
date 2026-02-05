//! Tree matching & evaluation — transpiled from lidaRtRee R source.
//!
//! Direct Rust port of `tree_matching()`, `hist_detection()`, and
//! `height_regression()`.

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A 3D point: (x, y, height).
pub type Point3D = [f64; 3];

/// A matched pair of reference and detected trees.
#[derive(Debug, Clone)]
pub struct MatchedPair {
    /// 0-based index into the reference array.
    pub r_idx: usize,
    /// 0-based index into the detected array.
    pub d_idx: usize,
    /// Height difference (detected − reference).
    pub h_diff: f64,
    /// Planimetric (2D) distance between the pair.
    pub plan_diff: f64,
}

/// Detection statistics.
#[derive(Debug, Clone)]
pub struct DetectionStats {
    pub true_detections: usize,
    pub false_detections: usize,
    pub omissions: usize,
}

/// Linear regression result for height comparison.
#[derive(Debug, Clone)]
pub struct HeightRegressionResult {
    /// Intercept of the fitted line `H_ref = intercept + slope * H_det`.
    pub intercept: f64,
    /// Slope.
    pub slope: f64,
    /// Root mean square error.
    pub rmse: f64,
    /// Bias (mean of `H_det − H_ref`).
    pub bias: f64,
    /// Standard deviation of `H_det − H_ref`.
    pub sd: f64,
}

// ---------------------------------------------------------------------------
// tree_matching
// ---------------------------------------------------------------------------

/// 3D matching of detected tree-top positions with reference positions.
///
/// First computes a matching index for each potential pair associating a
/// detected with a reference tree. This index is the 3D distance² between
/// detected and reference points, divided by a maximum matching distance²
/// set by `delta_ground` and `h_prec`.
///
/// Pairs with the lowest index are then iteratively associated (greedy).
///
/// # Arguments
/// - `lr` — reference positions `[x, y, h]`
/// - `ld` — detected positions `[x, y, h]`
/// - `delta_ground` — absolute ground buffer (m), default 2.1
/// - `h_prec` — height-proportional buffer, default 0.14
///
/// # Returns
/// Matched pairs sorted by matching index, or empty vec if none matched.
///
/// ## R equivalent
/// ```r
/// tree_matching(lr, ld, delta_ground = 2.1, h_prec = 0.14, stat = TRUE)
/// ```
pub fn tree_matching(
    lr: &[Point3D],
    ld: &[Point3D],
    delta_ground: f64,
    h_prec: f64,
) -> Vec<MatchedPair> {
    let nr = lr.len();
    let nd = ld.len();
    if nr == 0 || nd == 0 {
        return Vec::new();
    }

    // Coefficients for max matching squared radius:
    //   rmax = delta_ground + h_prec * H
    //   rmax² = delta_ground² + 2·h_prec·delta_ground·H + h_prec²·H²
    let d2max0 = delta_ground * delta_ground;
    let d2max1 = 2.0 * h_prec * delta_ground;
    let d2max2 = h_prec * h_prec;

    // norm_f[i] = max matching squared radius for reference tree i
    let norm_f: Vec<f64> = lr
        .iter()
        .map(|r| d2max0 + d2max1 * r[2] + d2max2 * r[2] * r[2])
        .collect();

    // Matrix of 3D squared distances (nd × nr)
    let mut d_sq: Vec<Vec<f64>> = vec![vec![0.0; nr]; nd];
    // Matrix of normalized matching indices
    let mut dn: Vec<Vec<f64>> = vec![vec![0.0; nr]; nd];

    for i in 0..nr {
        for j in 0..nd {
            let dx = ld[j][0] - lr[i][0];
            let dy = ld[j][1] - lr[i][1];
            let dz = ld[j][2] - lr[i][2];
            let dist2 = dx * dx + dy * dy + dz * dz;
            d_sq[j][i] = dist2;
            dn[j][i] = dist2 / norm_f[i];
        }
    }

    // Replace values ≥ 1.0 (over the matching limit) with 1.0
    for row in dn.iter_mut() {
        for v in row.iter_mut() {
            if *v >= 1.0 {
                *v = 1.0;
            }
        }
    }

    // Iterative greedy matching: pick the pair with the smallest index,
    // remove both the detected and reference tree from further matching.
    let mut matched: Vec<MatchedPair> = Vec::new();
    let mut used_r = vec![false; nr];
    let mut used_d = vec![false; nd];

    loop {
        // Find minimum value in dn that is < 1.0
        let mut min_val = 1.0_f64;
        let mut best_d = 0;
        let mut best_r = 0;

        for j in 0..nd {
            if used_d[j] {
                continue;
            }
            for i in 0..nr {
                if used_r[i] {
                    continue;
                }
                if dn[j][i] < min_val {
                    min_val = dn[j][i];
                    best_d = j;
                    best_r = i;
                }
            }
        }

        if min_val >= 1.0 {
            break; // No more pairs below the threshold
        }

        // Compute stats for this pair
        let h_diff = ld[best_d][2] - lr[best_r][2];
        let plan_diff = ((ld[best_d][0] - lr[best_r][0]).powi(2)
            + (ld[best_d][1] - lr[best_r][1]).powi(2))
        .sqrt();

        matched.push(MatchedPair {
            r_idx: best_r,
            d_idx: best_d,
            h_diff,
            plan_diff,
        });

        // Mark as used
        used_r[best_r] = true;
        used_d[best_d] = true;
    }

    matched
}

/// Convenience wrapper with default parameters (delta_ground=2.1, h_prec=0.14).
pub fn tree_matching_default(lr: &[Point3D], ld: &[Point3D]) -> Vec<MatchedPair> {
    tree_matching(lr, ld, 2.1, 0.14)
}

// ---------------------------------------------------------------------------
// hist_detection
// ---------------------------------------------------------------------------

/// Compute detection statistics: true detections, omissions, false detections.
///
/// ## R equivalent
/// ```r
/// hist_detection(lr, ld, matched, plot = FALSE)
/// ```
pub fn hist_detection(
    lr: &[Point3D],
    ld: &[Point3D],
    matched: &[MatchedPair],
) -> DetectionStats {
    let matched_r: std::collections::HashSet<usize> =
        matched.iter().map(|m| m.r_idx).collect();
    let matched_d: std::collections::HashSet<usize> =
        matched.iter().map(|m| m.d_idx).collect();

    let true_detections = matched.len();
    let omissions = lr.len() - matched_r.len();
    let false_detections = ld.len() - matched_d.len();

    DetectionStats {
        true_detections,
        false_detections,
        omissions,
    }
}

/// Compute height histograms for true detections, omissions, and false detections.
///
/// Returns (true_detection_heights, omission_heights, false_detection_heights).
pub fn hist_detection_heights(
    lr: &[Point3D],
    ld: &[Point3D],
    matched: &[MatchedPair],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let matched_r: std::collections::HashSet<usize> =
        matched.iter().map(|m| m.r_idx).collect();
    let matched_d: std::collections::HashSet<usize> =
        matched.iter().map(|m| m.d_idx).collect();

    let true_det: Vec<f64> = matched.iter().map(|m| lr[m.r_idx][2]).collect();
    let omissions: Vec<f64> = (0..lr.len())
        .filter(|i| !matched_r.contains(i))
        .map(|i| lr[i][2])
        .collect();
    let false_det: Vec<f64> = (0..ld.len())
        .filter(|i| !matched_d.contains(i))
        .map(|i| ld[i][2])
        .collect();

    (true_det, omissions, false_det)
}

// ---------------------------------------------------------------------------
// height_regression
// ---------------------------------------------------------------------------

/// Compute linear regression of reference heights vs detected heights.
///
/// Fits `H_ref = intercept + slope * H_det` and computes RMSE, bias, and SD.
///
/// ## R equivalent
/// ```r
/// height_regression(lr, ld, matched, plot = FALSE)
/// ```
pub fn height_regression(
    lr: &[Point3D],
    ld: &[Point3D],
    matched: &[MatchedPair],
) -> Option<HeightRegressionResult> {
    if matched.is_empty() {
        return None;
    }

    let n = matched.len() as f64;

    // Collect paired heights
    let h_ref: Vec<f64> = matched.iter().map(|m| lr[m.r_idx][2]).collect();
    let h_det: Vec<f64> = matched.iter().map(|m| ld[m.d_idx][2]).collect();

    // Simple linear regression: H_ref ~ H_det  →  y = a + b*x
    let mean_x: f64 = h_det.iter().sum::<f64>() / n;
    let mean_y: f64 = h_ref.iter().sum::<f64>() / n;

    let mut ss_xy = 0.0;
    let mut ss_xx = 0.0;
    for i in 0..matched.len() {
        let dx = h_det[i] - mean_x;
        let dy = h_ref[i] - mean_y;
        ss_xy += dx * dy;
        ss_xx += dx * dx;
    }

    let slope = if ss_xx.abs() > 1e-15 {
        ss_xy / ss_xx
    } else {
        0.0
    };
    let intercept = mean_y - slope * mean_x;

    // Compute error stats: differences = H_det - H_ref
    let diffs: Vec<f64> = h_det
        .iter()
        .zip(h_ref.iter())
        .map(|(d, r)| d - r)
        .collect();

    let rmse = (diffs.iter().map(|d| d * d).sum::<f64>() / n).sqrt();
    let bias = diffs.iter().sum::<f64>() / n;
    let sd = if n > 1.0 {
        let var = diffs.iter().map(|d| (d - bias).powi(2)).sum::<f64>() / (n - 1.0);
        var.sqrt()
    } else {
        0.0
    };

    Some(HeightRegressionResult {
        intercept,
        slope,
        rmse,
        bias,
        sd,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_matching_basic() {
        // Example from the R documentation
        let ref_trees: Vec<Point3D> = vec![
            [1.0, 1.0, 15.0],
            [4.0, 1.0, 18.0],
            [3.0, 2.0, 20.0],
            [4.0, 3.0, 10.0],
            [2.0, 4.0, 11.0],
        ];
        let det_trees: Vec<Point3D> = vec![
            [2.0, 1.0, 16.0],
            [2.0, 3.0, 19.0],
            [4.0, 4.0, 9.0],
            [4.0, 1.0, 15.0],
        ];

        let matched = tree_matching_default(&ref_trees, &det_trees);
        assert!(!matched.is_empty(), "Should find at least one match");

        // With default params, should match several pairs
        let stats = hist_detection(&ref_trees, &det_trees, &matched);
        assert_eq!(
            stats.true_detections + stats.omissions,
            ref_trees.len()
        );
        assert_eq!(
            stats.true_detections + stats.false_detections,
            det_trees.len()
        );
    }

    #[test]
    fn test_tree_matching_custom_params() {
        let ref_trees: Vec<Point3D> = vec![
            [1.0, 1.0, 15.0],
            [4.0, 1.0, 18.0],
            [3.0, 2.0, 20.0],
            [4.0, 3.0, 10.0],
            [2.0, 4.0, 11.0],
        ];
        let det_trees: Vec<Point3D> = vec![
            [2.0, 1.0, 16.0],
            [2.0, 3.0, 19.0],
            [4.0, 4.0, 9.0],
            [4.0, 1.0, 15.0],
        ];

        // Stricter matching
        let matched = tree_matching(&ref_trees, &det_trees, 2.0, 0.0);
        assert!(!matched.is_empty());
    }

    #[test]
    fn test_tree_matching_no_match() {
        let ref_trees: Vec<Point3D> = vec![[0.0, 0.0, 20.0]];
        let det_trees: Vec<Point3D> = vec![[100.0, 100.0, 20.0]];

        let matched = tree_matching_default(&ref_trees, &det_trees);
        assert!(matched.is_empty(), "Trees too far apart should not match");
    }

    #[test]
    fn test_height_regression() {
        let ref_trees: Vec<Point3D> = vec![
            [1.0, 1.0, 15.0],
            [4.0, 1.0, 18.0],
            [3.0, 2.0, 20.0],
        ];
        let det_trees: Vec<Point3D> = vec![
            [1.1, 1.0, 16.0],
            [4.0, 1.1, 19.0],
            [3.0, 2.1, 21.0],
        ];

        let matched = tree_matching_default(&ref_trees, &det_trees);
        let reg = height_regression(&ref_trees, &det_trees, &matched);
        assert!(reg.is_some());

        let reg = reg.unwrap();
        // Detected heights are ~1m higher → bias should be ~+1.0
        assert!((reg.bias - 1.0).abs() < 0.5);
        // Slope should be close to 1.0 (linear relationship)
        assert!((reg.slope - 1.0).abs() < 0.5);
    }

    #[test]
    fn test_hist_detection_counts() {
        let lr: Vec<Point3D> = vec![
            [0.0, 0.0, 10.0],
            [1.0, 0.0, 20.0],
            [2.0, 0.0, 30.0],
        ];
        let ld: Vec<Point3D> = vec![
            [0.1, 0.0, 11.0],
            [5.0, 5.0, 15.0],
        ];

        let matched = tree_matching_default(&lr, &ld);
        let stats = hist_detection(&lr, &ld, &matched);

        // Whatever the matching result, counts must be consistent
        assert_eq!(stats.true_detections + stats.omissions, lr.len());
    }
}
