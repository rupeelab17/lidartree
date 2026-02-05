//! Simple 2D raster grid, analogous to R's `SpatRaster` / `cimg`.

use std::ops::{Index, IndexMut};

/// A 2D raster (row-major). Origin is top-left, matching R's matrix convention.
///
/// Geo-referencing: pixel (r, c) maps to world coordinates:
///   x = xmin + (c + 0.5) * res_x
///   y = ymax - (r + 0.5) * res_y
#[derive(Debug, Clone)]
pub struct Raster {
    pub nrow: usize,
    pub ncol: usize,
    pub data: Vec<f64>,
    /// Resolution in X direction (cell width)
    pub res_x: f64,
    /// Resolution in Y direction (cell height)
    pub res_y: f64,
    /// Geographic extent
    pub xmin: f64,
    pub xmax: f64,
    pub ymin: f64,
    pub ymax: f64,
}

impl Raster {
    /// Create a new raster filled with a constant value.
    pub fn new(nrow: usize, ncol: usize, fill: f64) -> Self {
        Self {
            nrow,
            ncol,
            data: vec![fill; nrow * ncol],
            res_x: 1.0,
            res_y: 1.0,
            xmin: 0.0,
            xmax: ncol as f64,
            ymin: 0.0,
            ymax: nrow as f64,
        }
    }

    /// Create a raster from an existing Vec (row-major).
    pub fn from_vec(nrow: usize, ncol: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), nrow * ncol);
        Self {
            nrow,
            ncol,
            data,
            res_x: 1.0,
            res_y: 1.0,
            xmin: 0.0,
            xmax: ncol as f64,
            ymin: 0.0,
            ymax: nrow as f64,
        }
    }

    /// Create a raster with geographic extent.
    pub fn with_extent(mut self, xmin: f64, xmax: f64, ymin: f64, ymax: f64) -> Self {
        self.xmin = xmin;
        self.xmax = xmax;
        self.ymin = ymin;
        self.ymax = ymax;
        self.res_x = (xmax - xmin) / self.ncol as f64;
        self.res_y = (ymax - ymin) / self.nrow as f64;
        self
    }

    /// Get value at (row, col), returns NaN if out of bounds.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        if row < self.nrow && col < self.ncol {
            self.data[row * self.ncol + col]
        } else {
            f64::NAN
        }
    }

    /// Get value at (row, col) as Option.
    #[inline]
    pub fn get_opt(&self, row: isize, col: isize) -> Option<f64> {
        if row >= 0 && col >= 0 && (row as usize) < self.nrow && (col as usize) < self.ncol {
            let v = self.data[row as usize * self.ncol + col as usize];
            if v.is_nan() { None } else { Some(v) }
        } else {
            None
        }
    }

    /// Set value at (row, col).
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        if row < self.nrow && col < self.ncol {
            self.data[row * self.ncol + col] = val;
        }
    }

    /// Convert world X,Y to row,col.
    pub fn xy_to_rc(&self, x: f64, y: f64) -> (usize, usize) {
        let col = ((x - self.xmin) / self.res_x).floor() as usize;
        let row = ((self.ymax - y) / self.res_y).floor() as usize;
        (row.min(self.nrow.saturating_sub(1)), col.min(self.ncol.saturating_sub(1)))
    }

    /// Convert row,col to world X,Y (cell centre).
    pub fn rc_to_xy(&self, row: usize, col: usize) -> (f64, f64) {
        let x = self.xmin + (col as f64 + 0.5) * self.res_x;
        let y = self.ymax - (row as f64 + 0.5) * self.res_y;
        (x, y)
    }

    /// Number of cells.
    pub fn len(&self) -> usize {
        self.nrow * self.ncol
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Apply a function to every cell.
    pub fn map<F: Fn(f64) -> f64>(&self, f: F) -> Raster {
        let mut out = self.clone();
        for v in out.data.iter_mut() {
            *v = f(*v);
        }
        out
    }

    /// Pixel area in map units².
    pub fn cell_area(&self) -> f64 {
        self.res_x * self.res_y
    }
}

impl Index<(usize, usize)> for Raster {
    type Output = f64;
    fn index(&self, (r, c): (usize, usize)) -> &f64 {
        &self.data[r * self.ncol + c]
    }
}

impl IndexMut<(usize, usize)> for Raster {
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut f64 {
        &mut self.data[r * self.ncol + c]
    }
}
