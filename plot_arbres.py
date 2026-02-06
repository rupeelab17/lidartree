"""
Visualisation des arbres détectés par lidartree.

Lit le fichier arbres_detectes.csv et affiche les arbres sur le CDSM.

Usage :
    python plot_arbres.py arbres_detectes.csv CDSM_band0.tif
    python plot_arbres.py arbres_detectes.csv CDSM_band0.tif --hmin 10 --top 50 --save carte.png
"""

import argparse
import sys

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from shapely import wkt
from shapely.geometry import Point

# Optionnel : rasterio pour afficher le CDSM en fond
try:
    import rasterio
    from rasterio.plot import show as rioshow

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import tifffile

    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


def load_cdsm(path):
    """Charge un GeoTIFF mono-bande et retourne (array, extent)."""
    if HAS_RASTERIO:
        src = rasterio.open(path)
        data = src.read(1)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        return data, extent
    elif HAS_TIFFFILE:
        tif = tifffile.TiffFile(path)
        data = tif.pages[0].asarray()
        if data.ndim == 3:
            data = data[:, :, 0]
        # Lire les tags GeoTIFF
        tags = {t.code: t.value for t in tif.pages[0].tags.values()}
        if 33550 in tags and 33922 in tags:
            sx, sy = tags[33550][0], tags[33550][1]
            tx, ty = tags[33922][3], tags[33922][4]
            nrow, ncol = data.shape
            extent = [tx, tx + ncol * sx, ty - nrow * sy, ty]
        else:
            nrow, ncol = data.shape
            extent = [0, ncol, 0, nrow]
        return data.astype(np.float32), extent
    else:
        print("⚠ Ni rasterio ni tifffile installé — pas de fond raster.")
        return None, None


def load_trees(csv_path):
    """Charge le CSV et crée un GeoDataFrame."""
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} arbres chargés depuis '{csv_path}'")
    print(f"  Colonnes : {list(df.columns)}")

    # Créer la géométrie point
    geometry = [Point(xy) for xy in zip(df["x"], df["y"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:2154")

    # Si colonne crown_wkt existe, créer les polygones de couronnes
    if "crown_wkt" in df.columns:
        valid = df["crown_wkt"].notna() & (df["crown_wkt"] != "")
        if valid.any():
            crowns = df.loc[valid, "crown_wkt"].apply(wkt.loads)
            gdf_crowns = gpd.GeoDataFrame(
                df[valid], geometry=crowns.values, crs="EPSG:2154"
            )
            return gdf, gdf_crowns

    return gdf, None


def plot_trees(gdf, gdf_crowns, cdsm_data, cdsm_extent, args):
    """Crée la figure avec 4 sous-graphiques."""

    # Filtrer par hauteur min
    if args.hmin:
        gdf = gdf[gdf["h"] >= args.hmin].copy()
        if gdf_crowns is not None:
            gdf_crowns = gdf_crowns[gdf_crowns["h"] >= args.hmin].copy()
        print(f"  → {len(gdf)} arbres après filtre hmin={args.hmin}m")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(
        f"Détection d'arbres — {len(gdf)} arbres détectés",
        fontsize=16,
        fontweight="bold",
    )

    # ── 1. Carte des arbres sur le CDSM ─────────────────────────────
    ax1 = axes[0, 0]
    ax1.set_title("Arbres détectés sur le CDSM", fontsize=12)

    if cdsm_data is not None:
        masked = np.where(np.isnan(cdsm_data) | (cdsm_data <= 0), np.nan, cdsm_data)
        ax1.imshow(
            masked,
            extent=cdsm_extent,
            origin="upper",
            cmap="terrain",
            alpha=0.7,
            vmin=0,
            vmax=np.nanmax(masked),
        )

    sc1 = ax1.scatter(
        gdf["x"],
        gdf["y"],
        c=gdf["h"],
        s=gdf["h"] * 1.5,
        cmap="YlGn",
        edgecolors="black",
        linewidths=0.3,
        vmin=gdf["h"].min(),
        vmax=gdf["h"].max(),
        zorder=5,
    )
    plt.colorbar(sc1, ax=ax1, label="Hauteur (m)", shrink=0.8)
    ax1.set_xlabel("X (Lambert-93)")
    ax1.set_ylabel("Y (Lambert-93)")
    ax1.set_aspect("equal")

    # ── 2. Carte par surface de couronne ─────────────────────────────
    ax2 = axes[0, 1]
    ax2.set_title("Surface des couronnes", fontsize=12)

    if cdsm_data is not None:
        masked = np.where(np.isnan(cdsm_data) | (cdsm_data <= 0), np.nan, cdsm_data)
        ax2.imshow(
            masked,
            extent=cdsm_extent,
            origin="upper",
            cmap="Greys_r",
            alpha=0.4,
            vmin=0,
        )

    if gdf_crowns is not None and len(gdf_crowns) > 0:
        gdf_crowns.plot(
            ax=ax2,
            column="h",
            cmap="YlOrRd",
            edgecolor="black",
            linewidth=0.4,
            alpha=0.6,
            legend=True,
            legend_kwds={"label": "Hauteur apex (m)", "shrink": 0.8},
        )
    else:
        sc2 = ax2.scatter(
            gdf["x"],
            gdf["y"],
            c=gdf["surface"],
            s=gdf["surface"] * 0.5,
            cmap="YlOrRd",
            edgecolors="black",
            linewidths=0.3,
            zorder=5,
        )
        plt.colorbar(sc2, ax=ax2, label="Surface (m²)", shrink=0.8)

    ax2.set_xlabel("X (Lambert-93)")
    ax2.set_ylabel("Y (Lambert-93)")
    ax2.set_aspect("equal")

    # ── 3. Histogramme des hauteurs ──────────────────────────────────
    ax3 = axes[1, 0]
    ax3.set_title("Distribution des hauteurs", fontsize=12)

    bins = np.arange(0, gdf["h"].max() + 5, 5)
    counts, edges, patches = ax3.hist(
        gdf["h"], bins=bins, edgecolor="black", linewidth=0.5
    )

    # Colorier les barres par hauteur
    cmap = plt.cm.YlGn
    norm = mcolors.Normalize(vmin=edges[0], vmax=edges[-1])
    for patch, edge in zip(patches, edges[:-1]):
        patch.set_facecolor(cmap(norm(edge)))

    ax3.set_xlabel("Hauteur (m)")
    ax3.set_ylabel("Nombre d'arbres")
    ax3.axvline(
        gdf["h"].mean(),
        color="red",
        linestyle="--",
        label=f"Moyenne = {gdf['h'].mean():.1f}m",
    )
    ax3.axvline(
        gdf["h"].median(),
        color="blue",
        linestyle="--",
        label=f"Médiane = {gdf['h'].median():.1f}m",
    )
    ax3.legend()

    # ── 4. Top N arbres + stats ──────────────────────────────────────
    ax4 = axes[1, 1]
    ax4.set_title(f"Top {args.top} arbres les plus hauts", fontsize=12)

    top = gdf.nlargest(args.top, "h")
    bars = ax4.barh(
        range(len(top)),
        top["h"].values,
        color=cmap(norm(top["h"].values)),
        edgecolor="black",
        linewidth=0.3,
    )
    ax4.set_yticks(range(len(top)))
    ax4.set_yticklabels([f"#{i}" for i in top["id"].values], fontsize=7)
    ax4.set_xlabel("Hauteur (m)")
    ax4.set_ylabel("ID arbre")
    ax4.invert_yaxis()

    # Encart statistiques
    area_ha = (
        (gdf["x"].max() - gdf["x"].min()) * (gdf["y"].max() - gdf["y"].min()) / 10000
    )
    stats_text = (
        f"Arbres : {len(gdf)}\n"
        f"Surface : {area_ha:.1f} ha\n"
        f"Densité : {len(gdf) / max(area_ha, 0.01):.0f} /ha\n"
        f"H min : {gdf['h'].min():.1f} m\n"
        f"H moy : {gdf['h'].mean():.1f} m\n"
        f"H méd : {gdf['h'].median():.1f} m\n"
        f"H max : {gdf['h'].max():.1f} m\n"
        f"H σ   : {gdf['h'].std():.1f} m\n"
        f"Surf moy : {gdf['surface'].mean():.1f} m²"
    )
    ax4.text(
        0.95,
        0.95,
        stats_text,
        transform=ax4.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9),
        fontfamily="monospace",
    )

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"\n  → Carte sauvegardée : '{args.save}'")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualisation des arbres détectés par lidartree"
    )
    parser.add_argument("csv", help="Fichier CSV des arbres (arbres_detectes.csv)")
    parser.add_argument(
        "tif", nargs="?", default=None, help="CDSM GeoTIFF pour le fond (optionnel)"
    )
    parser.add_argument(
        "--hmin", type=float, default=None, help="Hauteur minimale à afficher (m)"
    )
    parser.add_argument(
        "--top", type=int, default=20, help="Nombre d'arbres dans le top (défaut: 20)"
    )
    parser.add_argument(
        "--save", type=str, default=None, help="Sauvegarder la figure (ex: carte.png)"
    )
    args = parser.parse_args()

    print("══════════════════════════════════════════════════")
    print("  Visualisation des arbres détectés — lidartree")
    print("══════════════════════════════════════════════════\n")

    # Charger les arbres
    gdf, gdf_crowns = load_trees(args.csv)

    # Charger le CDSM en fond
    cdsm_data, cdsm_extent = None, None
    if args.tif:
        print(f"\n  Chargement du raster '{args.tif}'...")
        cdsm_data, cdsm_extent = load_cdsm(args.tif)
        if cdsm_data is not None:
            print(f"  → {cdsm_data.shape[1]}×{cdsm_data.shape[0]} pixels")

    # Afficher
    print("\n  Génération de la carte...\n")
    plot_trees(gdf, gdf_crowns, cdsm_data, cdsm_extent, args)


if __name__ == "__main__":
    main()
