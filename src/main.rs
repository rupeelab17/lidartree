//! Détection d'arbres à partir d'un fichier CDSM GeoTIFF.
//!
//! Usage :
//!   cargo run --release -- CDSM.tif
//!   cargo run --release -- CDSM.tif --hmin 8 --sigma 0.8 --crown
//!
//! Produit :
//!   arbres_detectes.csv — id, x, y, h, dominance, surface, volume [, crown_wkt]

use lidartree::{
    raster::Raster,
    tree_detection::{tree_detection, DetectedTree, TreeSegmentationParams},
};
use std::fs::File;
use std::io::{BufReader, BufWriter};

use clap::Parser;
use csv::Writer;
use tiff::decoder::{Decoder, DecodingResult};
use tiff::tags::Tag;

// ==========================================================================
// Lecture des tags GeoTIFF via le crate tiff (ModelPixelScale + ModelTiepoint)
// ==========================================================================

/// Lit la résolution et l'origine géographique depuis les tags GeoTIFF du
/// décodeur TIFF déjà ouvert.
///
/// Retourne (res_x, res_y, origin_x, origin_y).
/// Utilise les tags ModelPixelScaleTag (33550) et ModelTiepointTag (33922).
fn geotiff_extent_from_decoder<R: std::io::Read + std::io::Seek>(
    decoder: &mut Decoder<R>,
) -> (f64, f64, f64, f64) {
    let mut res_x = 1.0_f64;
    let mut res_y = 1.0_f64;
    let mut origin_x = 0.0_f64;
    let mut origin_y = 0.0_f64;
    let mut found_scale = false;
    let mut found_tiepoint = false;

    if let Ok(Some(scale_val)) = decoder.find_tag(Tag::ModelPixelScaleTag) {
        if let Ok(scale) = scale_val.into_f64_vec() {
            if scale.len() >= 2 {
                res_x = scale[0];
                res_y = scale[1];
                found_scale = true;
            }
        }
    }
    if let Ok(Some(tie_val)) = decoder.find_tag(Tag::ModelTiepointTag) {
        if let Ok(tie) = tie_val.into_f64_vec() {
            if tie.len() >= 6 {
                origin_x = tie[3];
                origin_y = tie[4];
                found_tiepoint = true;
            }
        }
    }

    if !found_scale {
        eprintln!("⚠ Tag ModelPixelScale (33550) absent — résolution par défaut 1.0m");
    }
    if !found_tiepoint {
        eprintln!("⚠ Tag ModelTiepoint (33922) absent — origine par défaut (0, 0)");
    }

    println!(
        "  GeoTIFF tags : res=({}, {}), origin=({:.2}, {:.2})",
        res_x, res_y, origin_x, origin_y
    );

    (res_x, res_y, origin_x, origin_y)
}

// ==========================================================================
// Chargement GeoTIFF
// ==========================================================================

fn load_geotiff(path: &str) -> Raster {
    let file = File::open(path).unwrap_or_else(|e| {
        eprintln!("Impossible d'ouvrir '{}' : {}", path, e);
        std::process::exit(1);
    });
    let mut decoder = Decoder::new(BufReader::new(file)).unwrap_or_else(|e| {
        eprintln!("Erreur décodage TIFF : {}", e);
        std::process::exit(1);
    });

    let (width, height) = decoder.dimensions().unwrap();
    let ncol = width as usize;
    let nrow = height as usize;

    let (res_x, res_y, origin_x, origin_y) = geotiff_extent_from_decoder(&mut decoder);

    let xmin = origin_x;
    let ymax = origin_y;
    let xmax = xmin + ncol as f64 * res_x;
    let ymin = ymax - nrow as f64 * res_y;

    // Décoder les pixels
    let result = decoder.read_image().unwrap_or_else(|e| {
        eprintln!("Erreur lecture image : {}", e);
        std::process::exit(1);
    });

    let raw: Vec<f64> = match result {
        DecodingResult::F64(v) => v,
        DecodingResult::F32(v) => v.iter().map(|x| *x as f64).collect(),
        DecodingResult::U16(v) => v.iter().map(|x| *x as f64).collect(),
        DecodingResult::U8(v) => v.iter().map(|x| *x as f64).collect(),
        _ => {
            eprintln!("Format pixel non supporté");
            std::process::exit(1);
        }
    };

    // Multi-bande : le CDSM.tif a 3 bandes (interleaved).
    // Bande 0 = CDSM (6–62 m, avec NaN), bande 1 = vide, bande 2 = CHM normalisé.
    // On prend la bande 0 (altitudes de surface).
    let total_pixels = nrow * ncol;
    let n_bands = raw.len() / total_pixels;

    println!("  {} bande(s) détectée(s)", n_bands);

    let band_data: Vec<f64> = if n_bands > 1 {
        // Chercher la meilleure bande (max le plus élevé parmi celles avec >10% de pixels valides)
        let mut best_band = 0;
        let mut best_max = f64::NEG_INFINITY;
        for b in 0..n_bands {
            let mut bmax = f64::NEG_INFINITY;
            let mut valid_count = 0usize;
            for px in 0..total_pixels {
                let v = raw[px * n_bands + b];
                if !v.is_nan() && v > 0.0 {
                    valid_count += 1;
                    if v > bmax {
                        bmax = v;
                    }
                }
            }
            println!("    Bande {} : max={:.1}, valides={}", b, bmax, valid_count);
            if bmax > best_max && valid_count > total_pixels / 10 {
                best_max = bmax;
                best_band = b;
            }
        }
        println!("  → Bande {} sélectionnée", best_band);
        (0..total_pixels)
            .map(|px| raw[px * n_bands + best_band])
            .collect()
    } else {
        raw
    };

    Raster::from_vec(nrow, ncol, band_data).with_extent(xmin, xmax, ymin, ymax)
}

// ==========================================================================
// Export CSV
// ==========================================================================

fn export_csv(
    trees: &[DetectedTree],
    path: &str,
    with_crown: bool,
    source_per_tree: Option<&[String]>,
) {
    let file = File::create(path).unwrap_or_else(|e| {
        eprintln!("Impossible de créer '{}' : {}", path, e);
        std::process::exit(1);
    });
    let mut wtr = Writer::from_writer(BufWriter::new(file));

    let with_source = source_per_tree.is_some();
    let mut header: Vec<&str> = Vec::new();
    if with_source {
        header.push("source");
    }
    header.extend(["id", "x", "y", "h", "dom_radius", "surface", "volume"]);
    if with_crown {
        header.push("crown_wkt");
    }
    wtr.write_record(header).unwrap();

    for (i, t) in trees.iter().enumerate() {
        let id = t.id.to_string();
        let x = format!("{:.2}", t.x);
        let y = format!("{:.2}", t.y);
        let h = format!("{:.2}", t.h);
        let dom_radius = format!("{:.2}", t.dom_radius);
        let surface = format!("{:.1}", t.surface);
        let volume = format!("{:.1}", t.volume);
        let crown = t.crown_wkt.as_deref().unwrap_or("");

        let mut row: Vec<&str> = Vec::new();
        if let Some(sources) = source_per_tree {
            if i < sources.len() {
                row.push(sources[i].as_str());
            }
        }
        row.extend([
            id.as_str(),
            x.as_str(),
            y.as_str(),
            h.as_str(),
            dom_radius.as_str(),
            surface.as_str(),
            volume.as_str(),
        ]);
        if with_crown {
            row.push(crown);
        }
        wtr.write_record(row).unwrap();
    }

    wtr.flush().unwrap_or_else(|e| {
        eprintln!("Erreur flush CSV : {}", e);
        std::process::exit(1);
    });
}

// ==========================================================================
// CLI (clap)
// ==========================================================================

#[derive(Parser, Debug)]
#[command(
    name = "lidartree",
    about = "Détection d'arbres sur CDSM GeoTIFF",
    after_help = "Produit : arbres_detectes.csv — id, x, y, h, dominance, surface, volume [, crown_wkt]"
)]
struct Cli {
    /// Fichier(s) CDSM GeoTIFF en entrée (un ou plusieurs)
    #[arg(value_name = "CDSM.tif", num_args = 1..)]
    tif_paths: Vec<String>,

    /// Fichier MNT (DTM) optionnel : sélection/ajustement utilisent dem - dtm
    #[arg(long, value_name = "DTM.tif")]
    dtm: Option<String>,

    /// Masque raster optionnel : n'extraire que les arbres dont le sommet est dans le masque
    #[arg(long, value_name = "mask.tif")]
    mask: Option<String>,

    /// Hauteur min des arbres (m)
    #[arg(long, default_value_t = 5.0)]
    hmin: f64,

    /// Distance dominance min (m)
    #[arg(long, default_value_t = 0.5)]
    dmin: f64,

    /// Proportion dom/hauteur
    #[arg(long, default_value_t = 0.0)]
    dprop: f64,

    /// Sigma lissage gaussien
    #[arg(long, default_value_t = 0.6)]
    sigma: f64,

    /// Taille filtre médian
    #[arg(long, default_value_t = 3)]
    median: usize,

    /// Proportion base couronne (active le calcul de couronne si présent)
    #[arg(long, value_name = "f")]
    crown_prop: Option<f64>,

    /// Exporter les polygones couronne en WKT dans le CSV
    #[arg(long)]
    crown: bool,

    /// Couronne en ellipse (WKT) au lieu du convexe
    #[arg(long)]
    crown_ellipse: bool,

    /// Fichier CSV de sortie
    #[arg(long, short, default_value = "arbres_detectes.csv")]
    output: String,
}

// ==========================================================================
// Main
// ==========================================================================

fn main() {
    let cli = Cli::parse();

    if cli.tif_paths.is_empty() {
        eprintln!("Erreur : au moins un fichier CDSM.tif requis.");
        std::process::exit(1);
    }

    println!("══════════════════════════════════════════════════════════");
    println!("  lidartree — Détection d'arbres sur CDSM GeoTIFF");
    println!("══════════════════════════════════════════════════════════\n");

    let dtm_raster: Option<Raster> = cli.dtm.as_ref().map(|p| {
        println!("  Chargement DTM '{}'...", p);
        load_geotiff(p)
    });
    let dtm_ref = dtm_raster.as_ref();

    let mask_raster: Option<Raster> = cli.mask.as_ref().map(|p| {
        println!("  Chargement masque '{}'...", p);
        load_geotiff(p)
    });
    let mask_ref = mask_raster.as_ref();

    let params = TreeSegmentationParams {
        hmin: cli.hmin,
        dmin: cli.dmin,
        dprop: cli.dprop,
        nl_filter: "Median".into(),
        nl_size: cli.median,
        sigma: vec![(cli.sigma, 0.0)],
        crown_prop: cli.crown_prop,
        crown_hmin: cli.crown_prop.map(|_| 3.0),
        ..Default::default()
    };
    println!("2. Paramètres :");
    println!(
        "   hmin={:.1}m  dmin={:.1}m  dprop={:.2}  sigma={:.2}  median={}",
        cli.hmin, cli.dmin, cli.dprop, cli.sigma, cli.median
    );
    if cli.dtm.is_some() {
        println!("   DTM : oui (sélection/ajust sur dem-dtm)");
    }
    if cli.mask.is_some() {
        println!("   Masque : oui");
    }
    if let Some(cp) = cli.crown_prop {
        println!("   crown_prop={:.2}  couronne=oui", cp);
    }
    if cli.crown_ellipse {
        println!("   couronne : ellipse");
    }
    println!();

    let mut all_trees: Vec<DetectedTree> = Vec::new();
    let mut all_sources: Vec<String> = Vec::new();
    let mut total_ha = 0.0_f64;

    for tif_path in &cli.tif_paths {
        println!("1. Chargement de '{}'...", tif_path);
        let chm = load_geotiff(tif_path);
        println!(
            "   {} × {} pixels  ({:.0} × {:.0} m = {:.2} ha)",
            chm.ncol,
            chm.nrow,
            chm.xmax - chm.xmin,
            chm.ymax - chm.ymin,
            (chm.xmax - chm.xmin) * (chm.ymax - chm.ymin) / 10000.0
        );
        total_ha += (chm.xmax - chm.xmin) * (chm.ymax - chm.ymin) / 10000.0;

        if let Some(dtm) = dtm_ref {
            if dtm.nrow != chm.nrow || dtm.ncol != chm.ncol {
                eprintln!("   Erreur : le DTM doit avoir les mêmes dimensions que le CDSM.");
                std::process::exit(1);
            }
        }
        if let Some(mask) = mask_ref {
            if mask.nrow != chm.nrow || mask.ncol != chm.ncol {
                eprintln!("   Erreur : le masque doit avoir les mêmes dimensions que le CDSM.");
                std::process::exit(1);
            }
        }

        println!("3. Détection en cours...");
        let t0 = std::time::Instant::now();
        let trees = tree_detection(
            &chm,
            &params,
            cli.crown,
            dtm_ref,
            mask_ref,
            cli.crown_ellipse,
        );
        let dt = t0.elapsed();
        println!(
            "   ✓ {} arbres détectés en {:.2}s\n",
            trees.len(),
            dt.as_secs_f64()
        );

        let n = all_trees.len();
        all_trees.extend(trees);
        if cli.tif_paths.len() > 1 {
            let path = tif_path.clone();
            for _ in n..all_trees.len() {
                all_sources.push(path.clone());
            }
        }
    }

    // Statistiques (sur l'ensemble si un seul fichier, sinon agrégé)
    if !all_trees.is_empty() {
        let hs: Vec<f64> = all_trees.iter().map(|t| t.h).collect();
        let ss: Vec<f64> = all_trees.iter().map(|t| t.surface).collect();
        let h_min_d = hs.iter().cloned().fold(f64::INFINITY, f64::min);
        let h_max_d = hs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let h_mean_d = hs.iter().sum::<f64>() / hs.len() as f64;
        let s_mean = ss.iter().sum::<f64>() / ss.len() as f64;
        let s_tot = ss.iter().sum::<f64>();
        let density = all_trees.len() as f64 / total_ha;

        println!("4. Statistiques :");
        println!(
            "   Hauteurs    : min={:.1}  moy={:.1}  max={:.1} m",
            h_min_d, h_mean_d, h_max_d
        );
        println!("   Surf. moy.  : {:.1} m²/arbre", s_mean);
        if total_ha > 0.0 {
            println!(
                "   Couvert     : {:.1}% de la zone",
                s_tot / (total_ha * 10000.0) * 100.0
            );
            println!("   Densité     : {:.0} arbres/ha", density);
        }

        let mut sorted = all_trees.clone();
        sorted.sort_by(|a, b| b.h.partial_cmp(&a.h).unwrap());
        println!("\n   Top 10 arbres les plus hauts :");
        println!(
            "   {:>4}  {:>12}  {:>12}  {:>6}  {:>8}  {:>8}",
            "ID", "X (L93)", "Y (L93)", "H (m)", "Surf m²", "Vol m³"
        );
        println!("   {}", "─".repeat(60));
        for t in sorted.iter().take(10) {
            println!(
                "   {:>4}  {:>12.2}  {:>12.2}  {:>6.1}  {:>8.1}  {:>8.1}",
                t.id, t.x, t.y, t.h, t.surface, t.volume
            );
        }
    }

    println!("\n5. Export → '{}'", cli.output);
    let with_source = cli.tif_paths.len() > 1 && !all_sources.is_empty();
    export_csv(&all_trees, &cli.output, cli.crown, with_source.then(|| all_sources.as_slice()));
    println!("   {} lignes écrites.", all_trees.len());

    println!("\n══════════════════════════════════════════════════════════");
    println!(
        "  ✓ Terminé — {} arbres sur {:.2} ha",
        all_trees.len(),
        total_ha
    );
    println!("══════════════════════════════════════════════════════════");
}
