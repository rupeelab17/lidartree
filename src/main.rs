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
use std::env;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};

use tiff::decoder::{Decoder, DecodingResult};

// ==========================================================================
// Lecture des tags GeoTIFF (ModelPixelScale + ModelTiepoint)
// ==========================================================================

/// Parse les tags GeoTIFF directement depuis le fichier TIFF pour extraire
/// la résolution et l'origine géographique.
///
/// Retourne (res_x, res_y, origin_x, origin_y).
///
/// Tags recherchés :
///   - 33550 (ModelPixelScaleTag)  : 3 doubles → (scaleX, scaleY, scaleZ)
///   - 33922 (ModelTiepointTag)    : 6 doubles → (pixCol, pixRow, pixZ, geoX, geoY, geoZ)
fn read_geotiff_tags(path: &str) -> (f64, f64, f64, f64) {
    let mut res_x = 1.0_f64;
    let mut res_y = 1.0_f64;
    let mut origin_x = 0.0_f64;
    let mut origin_y = 0.0_f64;

    let mut file = File::open(path).unwrap_or_else(|e| {
        eprintln!("Impossible d'ouvrir '{}' : {}", path, e);
        std::process::exit(1);
    });

    // Lire le header TIFF pour déterminer l'endianness et l'offset de l'IFD
    let mut header = [0u8; 8];
    file.read_exact(&mut header).unwrap();

    let little_endian = match &header[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => {
            eprintln!("⚠ Pas un fichier TIFF valide, utilisation de coordonnées par défaut");
            return (res_x, res_y, origin_x, origin_y);
        }
    };

    let read_u16 = |buf: &[u8]| -> u16 {
        if little_endian {
            u16::from_le_bytes([buf[0], buf[1]])
        } else {
            u16::from_be_bytes([buf[0], buf[1]])
        }
    };
    let read_u32 = |buf: &[u8]| -> u32 {
        if little_endian {
            u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]])
        } else {
            u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]])
        }
    };
    let read_f64 = |buf: &[u8]| -> f64 {
        if little_endian {
            f64::from_le_bytes([
                buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
            ])
        } else {
            f64::from_be_bytes([
                buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
            ])
        }
    };

    // Offset du premier IFD
    let ifd_offset = read_u32(&header[4..8]);
    file.seek(SeekFrom::Start(ifd_offset as u64)).unwrap();

    // Nombre d'entrées dans l'IFD
    let mut buf2 = [0u8; 2];
    file.read_exact(&mut buf2).unwrap();
    let num_entries = read_u16(&buf2);

    let mut found_scale = false;
    let mut found_tiepoint = false;

    for _ in 0..num_entries {
        let mut entry = [0u8; 12];
        file.read_exact(&mut entry).unwrap();

        let tag_id = read_u16(&entry[0..2]);
        let _type_id = read_u16(&entry[2..4]);
        let count = read_u32(&entry[4..8]);
        let value_offset = read_u32(&entry[8..12]);

        match tag_id {
            33550 => {
                // ModelPixelScaleTag : 3 doubles (scaleX, scaleY, scaleZ)
                let pos = file.stream_position().unwrap();
                file.seek(SeekFrom::Start(value_offset as u64)).unwrap();
                let mut dbuf = [0u8; 24];
                file.read_exact(&mut dbuf).unwrap();
                res_x = read_f64(&dbuf[0..8]);
                res_y = read_f64(&dbuf[8..16]);
                // dbuf[16..24] = scaleZ (ignoré)
                found_scale = true;
                file.seek(SeekFrom::Start(pos)).unwrap();
            }
            33922 => {
                // ModelTiepointTag : 6 doubles (pixCol, pixRow, pixZ, geoX, geoY, geoZ)
                let pos = file.stream_position().unwrap();
                file.seek(SeekFrom::Start(value_offset as u64)).unwrap();
                let n_bytes = (count as usize) * 8;
                let mut dbuf = vec![0u8; n_bytes];
                file.read_exact(&mut dbuf).unwrap();
                // On lit le premier tiepoint (il peut y en avoir plusieurs)
                let _pix_col = read_f64(&dbuf[0..8]);
                let _pix_row = read_f64(&dbuf[8..16]);
                // dbuf[16..24] = pixZ
                origin_x = read_f64(&dbuf[24..32]);
                origin_y = read_f64(&dbuf[32..40]);
                // dbuf[40..48] = geoZ
                found_tiepoint = true;
                file.seek(SeekFrom::Start(pos)).unwrap();
            }
            _ => {}
        }

        if found_scale && found_tiepoint {
            break;
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

    // Emprise GeoTIFF — lue depuis les tags inspectés :
    //   ModelPixelScaleTag  (33550) : (1.0, 1.0, 0.0)
    //   ModelTiepointTag    (33922) : (0,0,0, 379818.38, 6573607.81, 0)
    //   CRS : EPSG:2154 (RGF93 / Lambert-93)
    let (res_x, res_y, origin_x, origin_y) = read_geotiff_tags(path);

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

fn export_csv(trees: &[DetectedTree], path: &str, with_crown: bool) {
    let file = File::create(path).unwrap();
    let mut w = BufWriter::new(file);

    if with_crown {
        writeln!(w, "id,x,y,h,dom_radius,surface,volume,crown_wkt").unwrap();
    } else {
        writeln!(w, "id,x,y,h,dom_radius,surface,volume").unwrap();
    }
    for t in trees {
        if with_crown {
            writeln!(
                w,
                "{},{:.2},{:.2},{:.2},{:.2},{:.1},{:.1},\"{}\"",
                t.id,
                t.x,
                t.y,
                t.h,
                t.dom_radius,
                t.surface,
                t.volume,
                t.crown_wkt.as_deref().unwrap_or("")
            )
            .unwrap();
        } else {
            writeln!(
                w,
                "{},{:.2},{:.2},{:.2},{:.2},{:.1},{:.1}",
                t.id, t.x, t.y, t.h, t.dom_radius, t.surface, t.volume
            )
            .unwrap();
        }
    }
}

// ==========================================================================
// Main
// ==========================================================================

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage : {} <CDSM.tif> [options]\n", args[0]);
        eprintln!("Options :");
        eprintln!("  --hmin <m>       Hauteur min des arbres    (défaut: 5.0)");
        eprintln!("  --dmin <m>       Distance dominance min    (défaut: 0.5)");
        eprintln!("  --dprop <f>      Proportion dom/hauteur    (défaut: 0.0)");
        eprintln!("  --sigma <f>      Sigma lissage gaussien    (défaut: 0.6)");
        eprintln!("  --median <n>     Taille filtre médian      (défaut: 3)");
        eprintln!("  --crown-prop <f> Proportion base couronne");
        eprintln!("  --crown          Exporter polygones WKT");
        eprintln!("  --output <f>     Fichier CSV sortie        (défaut: arbres_detectes.csv)");
        std::process::exit(1);
    }

    println!("══════════════════════════════════════════════════════════");
    println!("  lidartree — Détection d'arbres sur CDSM GeoTIFF");
    println!("══════════════════════════════════════════════════════════\n");

    let tif_path = &args[1];
    let mut hmin = 5.0_f64;
    let mut dmin = 0.5_f64;
    let mut dprop = 0.0_f64;
    let mut sigma = 0.6_f64;
    let mut median_size = 3_usize;
    let mut crown_prop: Option<f64> = None;
    let mut compute_crown = false;
    let mut output_csv = "arbres_detectes.csv".to_string();

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--hmin" => {
                hmin = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--dmin" => {
                dmin = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--dprop" => {
                dprop = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--sigma" => {
                sigma = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--median" => {
                median_size = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--crown-prop" => {
                crown_prop = Some(args[i + 1].parse().unwrap());
                i += 2;
            }
            "--crown" => {
                compute_crown = true;
                i += 1;
            }
            "--output" => {
                output_csv = args[i + 1].clone();
                i += 2;
            }
            other => {
                eprintln!("Option inconnue : {}", other);
                std::process::exit(1);
            }
        }
    }

    // ── 1. Charger le CDSM ──────────────────────────────────────────────
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
    println!(
        "   Emprise : X [{:.2} → {:.2}]  Y [{:.2} → {:.2}]",
        chm.xmin, chm.xmax, chm.ymin, chm.ymax
    );
    println!("   CRS     : EPSG:2154 (RGF93 / Lambert-93)");

    let valid: Vec<f64> = chm
        .data
        .iter()
        .copied()
        .filter(|v| !v.is_nan() && *v > 0.0)
        .collect();
    let h_max = valid.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let h_mean = valid.iter().sum::<f64>() / valid.len() as f64;
    println!(
        "   Hauteurs: moy={:.1}m  max={:.1}m  ({} px valides)\n",
        h_mean,
        h_max,
        valid.len()
    );

    // ── 2. Paramètres ───────────────────────────────────────────────────
    let params = TreeSegmentationParams {
        hmin,
        dmin,
        dprop,
        nl_filter: "Median".into(),
        nl_size: median_size,
        sigma: vec![(sigma, 0.0)],
        crown_prop,
        crown_hmin: crown_prop.map(|_| 3.0),
        ..Default::default()
    };
    println!("2. Paramètres :");
    println!(
        "   hmin={:.1}m  dmin={:.1}m  dprop={:.2}  sigma={:.2}  median={}",
        hmin, dmin, dprop, sigma, median_size
    );
    if let Some(cp) = crown_prop {
        println!("   crown_prop={:.2}  couronne=oui", cp);
    }
    println!();

    // ── 3. Détection ────────────────────────────────────────────────────
    println!("3. Détection en cours...");
    let t0 = std::time::Instant::now();
    let trees = tree_detection(&chm, &params, compute_crown);
    let dt = t0.elapsed();
    println!(
        "   ✓ {} arbres détectés en {:.2}s\n",
        trees.len(),
        dt.as_secs_f64()
    );

    // ── 4. Statistiques ─────────────────────────────────────────────────
    if !trees.is_empty() {
        let hs: Vec<f64> = trees.iter().map(|t| t.h).collect();
        let ss: Vec<f64> = trees.iter().map(|t| t.surface).collect();
        let h_min_d = hs.iter().cloned().fold(f64::INFINITY, f64::min);
        let h_max_d = hs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let h_mean_d = hs.iter().sum::<f64>() / hs.len() as f64;
        let s_mean = ss.iter().sum::<f64>() / ss.len() as f64;
        let s_tot = ss.iter().sum::<f64>();
        let area_ha = (chm.xmax - chm.xmin) * (chm.ymax - chm.ymin) / 10000.0;
        let density = trees.len() as f64 / area_ha;

        println!("4. Statistiques :");
        println!(
            "   Hauteurs    : min={:.1}  moy={:.1}  max={:.1} m",
            h_min_d, h_mean_d, h_max_d
        );
        println!("   Surf. moy.  : {:.1} m²/arbre", s_mean);
        println!(
            "   Couvert     : {:.1}% de la zone",
            s_tot / (area_ha * 10000.0) * 100.0
        );
        println!("   Densité     : {:.0} arbres/ha", density);

        // Histogramme par classes
        println!("\n   Classes de hauteur :");
        for &(lo, hi) in &[
            (0.0, 10.0),
            (10.0, 20.0),
            (20.0, 30.0),
            (30.0, 40.0),
            (40.0, 50.0),
            (50.0, 100.0),
        ] {
            let n = hs.iter().filter(|h| **h >= lo && **h < hi).count();
            if n > 0 {
                let bar = "█".repeat(((n as f64 / trees.len() as f64) * 50.0).ceil() as usize);
                println!("   {:>3.0}–{:<3.0}m : {:>5}  {}", lo, hi, n, bar);
            }
        }

        // Top 10
        let mut sorted = trees.clone();
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

    // ── 5. Export ───────────────────────────────────────────────────────
    println!("\n5. Export → '{}'", output_csv);
    export_csv(&trees, &output_csv, compute_crown);
    println!("   {} lignes écrites.", trees.len());

    println!("\n══════════════════════════════════════════════════════════");
    println!(
        "  ✓ Terminé — {} arbres sur {:.2} ha",
        trees.len(),
        (chm.xmax - chm.xmin) * (chm.ymax - chm.ymin) / 10000.0
    );
    println!("══════════════════════════════════════════════════════════");
}
