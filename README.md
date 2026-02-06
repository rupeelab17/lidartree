# lidartree
Lidar Tree Detection 

# 🌲 lidartree

**Rust port of the R package [lidaRtRee](https://cran.r-project.org/package=lidaRtRee)** — Forest analysis with Airborne Laser Scanning (LiDAR) data.

Transpiled from the original R code by Jean-Matthieu Monnet (INRAE), GPL-3.

## Features

- **Tree detection pipeline** on Canopy Height Models (CHM / CDSM):
  - Median filtering + Gaussian smoothing (`dem_filtering`)
  - Variable-window local maxima detection (`maxima_detection`)
  - Maxima selection by height & dominance (`maxima_selection`)
  - Seeded watershed segmentation (`segmentation`)
  - Crown base adjustment (`seg_adjust`)
  - Full pipeline: `tree_segmentation` → `tree_extraction` → `tree_detection`
- **Tree matching & evaluation**:
  - 3D matching of detected vs reference trees (`tree_matching`)
  - Detection statistics: TP / FP / FN (`hist_detection`)
  - Height regression: RMSE, bias, slope (`height_regression`)
- **Python visualization** script (`plot_arbres.py`)

## Quick start

```bash
# Build
cargo build --release

# Run on a GeoTIFF CHM (mono-band Float32)
cargo run --release -- CDSM_f32.tif

# With custom parameters
cargo run --release -- CDSM_f32.tif --hmin 8 --sigma 0.8 --dmin 1.0

# Export crown polygons as WKT
cargo run --release -- CDSM_f32.tif --hmin 8 --crown --output arbres.csv
```

### Input format

The input must be a **single-band Float32 GeoTIFF**. If your CDSM is multi-band Float64, convert it first:

```bash
gdal_translate -b 1 -ot Float32 CDSM.tif CDSM_f32.tif
```

### Output

`arbres_detectes.csv` with columns: `id, x, y, h, dom_radius, surface, volume [, crown_wkt]`

Coordinates are in the same CRS as the input raster (e.g. EPSG:2154 Lambert-93).

## Visualization (Python)

```bash
pip install geopandas matplotlib shapely pandas tifffile

# Plot detected trees on the CDSM
python plot_arbres.py arbres_detectes.csv CDSM_f32.tif

# Filter + save
python plot_arbres.py arbres_detectes.csv CDSM_f32.tif --hmin 10 --save carte.png
```

Produces 4 panels: tree map on CDSM, crown surfaces, height histogram, top-N trees + statistics.

## CLI options

| Option | Default | Description |
|---|---|---|
| `--dtm <DTM.tif>` | — | Optional DTM: selection/seg_adjust use dem − dtm (canopy height) |
| `--mask <mask.tif>` | — | Optional raster mask: only extract trees whose apex is inside mask (R: `r_mask`) |
| `--hmin <m>` | 5.0 | Minimum tree height (m) |
| `--dmin <m>` | 0.5 | Minimum dominance distance (m) |
| `--dprop <f>` | 0.0 | Dominance as proportion of height |
| `--sigma <f>` | 0.6 | Gaussian smoothing sigma |
| `--median <n>` | 3 | Median filter kernel size (odd) |
| `--crown-prop <f>` | — | Crown base height as proportion of apex |
| `--crown` | off | Compute crown WKT polygons |
| `--crown-ellipse` | off | Crown as ellipse WKT (R: ellipses4Crown-style) instead of convex hull |
| `-o, --output <f>` | arbres_detectes.csv | Output CSV path |

You can pass **multiple CDSM files**; the CSV will include a `source` column with the file path.

## Project structure

```
src/
├── lib.rs               # Library entry point
├── raster.rs            # 2D georeferenced raster type
├── tree_detection.rs    # Detection pipeline (maxima, watershed, segmentation)
├── tree_matching.rs     # Tree matching & evaluation
└── main.rs              # CLI: load GeoTIFF → detect → export CSV
plot_arbres.py           # Python visualization script
```

## Documentation R (référence)

- **Package** : [lidaRtRee sur CRAN](https://cran.r-project.org/package=lidaRtRee) (v4.0.x).
- **Vignette** : [Tree segmentation](https://lidar.pages.mia.inra.fr/lidaRtRee/articles/tree.detection.html) — workflow treetop detection, segmentation, évaluation avec inventaire.
- **Dépôt** : [forgemia.inra.fr/lidar/lidaRtRee](https://forgemia.inra.fr/lidar/lidaRtRee).

## References

- Monnet, J.-M. 2011. *Using airborne laser scanning for mountain forests mapping*. Ph.D. thesis, University of Grenoble. [PDF](https://theses.hal.science/tel-00652698/document)
- Eysn et al. 2015. A benchmark of lidar-based single tree detection methods. *Forests* 6(5). [doi:10.3390/f6051721](https://doi.org/10.3390/f6051721)

## License

GPL-3.0 (same as original lidaRtRee R package)
