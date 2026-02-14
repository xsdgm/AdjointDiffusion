"""
PBS Structure Post-Processing: Vectorization & Smoothing (Paper Polishing)

Pipeline:
  1. Morphological open/close  — remove islands & fill small holes
  2. Connected-component filter — drop regions smaller than `min_area_px`
  3. Gaussian blur + re-threshold — smooth edges while preserving topology
  4. Contour extraction + spline smoothing — for vectorized output (SVG/GDS)
  5. Export comparison figure + polished GDS

Note: Step 3 produces the polished raster for re-simulation.
      Step 4 is ONLY for visualization / GDSII export, NOT for rasterization
      (contour-based rasterization has topology issues with holes).
"""

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.interpolate import splprep, splev
from skimage import measure, morphology

try:
    import gdstk
except Exception:
    gdstk = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _normalize(raw: np.ndarray) -> np.ndarray:
    arr = raw.astype("float32")
    if float(arr.max()) > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _morphological_clean(
    binary: np.ndarray,
    open_r: int = 0,
    close_r: int = 1,
) -> np.ndarray:
    """Apply morphological opening then closing to remove islands & spurs."""
    if open_r > 0:
        selem_open = morphology.disk(open_r)
        binary = morphology.binary_opening(binary, selem_open)
    if close_r > 0:
        selem_close = morphology.disk(close_r)
        binary = morphology.binary_closing(binary, selem_close)
    return binary


def _remove_small_features(
    binary: np.ndarray,
    min_area_px: int = 8,
) -> np.ndarray:
    """Remove connected components (both Si and air) smaller than min_area_px."""
    # remove small Si islands
    labeled_si, n_si = ndimage.label(binary)
    for i in range(1, n_si + 1):
        if np.sum(labeled_si == i) < min_area_px:
            binary[labeled_si == i] = 0

    # remove small air holes (invert, clean, invert back)
    inv = ~binary
    labeled_air, n_air = ndimage.label(inv)
    for i in range(1, n_air + 1):
        if np.sum(labeled_air == i) < min_area_px:
            binary[labeled_air == i] = 1

    return binary


def _gaussian_smooth_binary(
    binary: np.ndarray,
    sigma: float = 0.8,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Smooth binary image edges using Gaussian blur + re-threshold.

    This method preserves topology (no holes are filled/lost) while rounding
    off jagged pixel-staircase edges. Much safer than contour-based
    rasterization which can invert hole regions.

    Parameters
    ----------
    sigma : float
        Gaussian blur sigma in pixels. Larger = smoother but more topological
        change risk. 0.5-1.0 is a safe range for 64x64 grids.
    threshold : float
        Re-binarization threshold after blurring (0.5 = neutral).
    """
    blurred = ndimage.gaussian_filter(binary.astype("float64"), sigma=sigma)
    return blurred >= threshold


def _smooth_contours(
    binary: np.ndarray,
    level: float = 0.5,
    smoothing_factor: float = 2.0,
    num_points: int = 300,
    min_contour_len: int = 10,
) -> List[np.ndarray]:
    """
    Extract contours from binary image and smooth them with B-spline.
    Used ONLY for vectorized visualization (SVG/GDS), NOT for rasterization.

    Returns
    -------
    smoothed : list of (N, 2) arrays  – each row is (row, col) in pixel coords
    """
    contours = measure.find_contours(binary.astype(float), level)
    smoothed = []
    for c in contours:
        if len(c) < min_contour_len:
            continue
        # ensure closed
        if not np.allclose(c[0], c[-1], atol=0.5):
            c = np.vstack([c, c[:1]])

        try:
            tck, u = splprep([c[:, 0], c[:, 1]], s=smoothing_factor, per=True)
            u_new = np.linspace(0, 1, num_points)
            row_s, col_s = splev(u_new, tck)
            # clip to valid pixel range
            row_s = np.clip(row_s, 0, binary.shape[0] - 1)
            col_s = np.clip(col_s, 0, binary.shape[1] - 1)
            smoothed.append(np.column_stack([row_s, col_s]))
        except Exception:
            # fall-back: keep original
            smoothed.append(c)
    return smoothed


# ---------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------

def polish_pbs_structure(
    npz_path: str,
    out_dir: str,
    sample_index: int = 0,
    threshold: float = 0.5,
    design_size_um: float = 3.0,
    # morphology params
    open_radius: int = 0,
    close_radius: int = 1,
    min_area_px: int = 8,
    # Gaussian smoothing params (for raster output)
    gauss_sigma: float = 0.8,
    # contour smoothing params (for vector output only)
    smoothing_factor: float = 2.0,
    num_points: int = 300,
    min_contour_len: int = 10,
    # GDS params
    gds_layer: int = 1,
    gds_datatype: int = 0,
) -> None:
    # ---- load ----------------------------------------------------------
    npz = np.load(npz_path)
    raw = npz["arr_0"][sample_index, :, :, 0]
    struct = _normalize(raw)
    binary_orig = (struct > threshold).astype(bool)

    # ---- step 1: morphology clean (gentle) ----------------------------
    cleaned = _morphological_clean(
        binary_orig.copy(), open_r=open_radius, close_r=close_radius
    )

    # ---- step 2: small feature removal --------------------------------
    cleaned = _remove_small_features(cleaned.copy(), min_area_px=min_area_px)

    # ---- step 3: Gaussian smooth → polished raster (for simulation) ----
    polished = _gaussian_smooth_binary(cleaned, sigma=gauss_sigma, threshold=threshold)

    # ---- step 4: contour smoothing (for vector output only) -----------
    smooth_contours = _smooth_contours(
        polished,
        smoothing_factor=smoothing_factor,
        num_points=num_points,
        min_contour_len=min_contour_len,
    )

    # ---- 统计 -------------------------------------------------------
    n_orig = int(binary_orig.sum())
    n_pol  = int(polished.sum())
    diff   = (polished.astype(int) - binary_orig.astype(int))
    n_same = int(np.sum(diff == 0))
    total  = binary_orig.size
    print(f"像素统计: 原始 Si={n_orig}, 平滑 Si={n_pol}, "
          f"不变={n_same}/{total} ({100*n_same/total:.1f}%)")

    # ---- output --------------------------------------------------------
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- save polished binary as .npz so pbs_metrics/eval can re-evaluate ---
    polished_uint8 = (polished.astype("float32") * 255).astype("uint8")
    polished_npz = polished_uint8[np.newaxis, :, :, np.newaxis]  # (1, Nx, Ny, 1)
    npz_out_path = out_dir / "polished_structure.npz"
    np.savez(npz_out_path, polished_npz)

    # --- comparison figure ---
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    nx, ny = binary_orig.shape

    ax = axes[0]
    ax.imshow(binary_orig.T.astype(float), cmap="gray", origin="lower", vmin=0, vmax=1)
    ax.set_title("Original Binary")

    ax = axes[1]
    ax.imshow(cleaned.T.astype(float), cmap="gray", origin="lower", vmin=0, vmax=1)
    ax.set_title("After Morphology Clean")

    ax = axes[2]
    ax.imshow(polished.T.astype(float), cmap="gray", origin="lower", vmin=0, vmax=1)
    # overlay smooth contours
    for c in smooth_contours:
        # contours are (row, col); imshow with .T maps row→x, col→y
        ax.plot(c[:, 0], c[:, 1], color="cyan", linewidth=0.6, alpha=0.8)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_title("Polished (Gaussian Smooth)")

    for a in axes:
        a.set_aspect("equal")
    plt.suptitle("PBS Structure — Paper Polishing Pipeline", fontsize=12)
    plt.tight_layout()
    cmp_path = out_dir / "pbs_polish_comparison.png"
    plt.savefig(cmp_path, dpi=300)
    plt.close()

    # --- standalone polished image ---
    plt.figure(figsize=(4.2, 4.2))
    plt.imshow(polished.T.astype(float), cmap="gray", origin="lower", vmin=0, vmax=1)
    plt.colorbar(label="Binary")
    plt.title("PBS Structure (Polished)")
    plt.tight_layout()
    pol_img_path = out_dir / "pbs_structure_polished.png"
    plt.savefig(pol_img_path, dpi=300)
    plt.close()

    # --- contour-only vector figure (SVG + PNG) ---
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    # fill background black (SiO2), contours white (Si) — match original convention
    ax.set_facecolor("black")
    for c in smooth_contours:
        ax.fill(c[:, 0], c[:, 1], color="white", alpha=1.0)
        ax.plot(c[:, 0], c[:, 1], color="white", linewidth=0.5)
    ax.set_aspect("equal")
    ax.set_title("Vectorized PBS Contours")
    plt.tight_layout()
    svg_path = out_dir / "pbs_structure_vector.svg"
    png_vec_path = out_dir / "pbs_structure_vector.png"
    plt.savefig(svg_path)
    plt.savefig(png_vec_path, dpi=300)
    plt.close()

    # --- polished GDS ---
    if gdstk is not None:
        gds_path = out_dir / "pbs_structure_polished.gds"
        _save_smooth_gds(
            smooth_contours,
            binary_orig.shape,
            design_size_um,
            gds_path,
            gds_layer,
            gds_datatype,
        )
        print(gds_path.as_posix())
    else:
        print("[WARN] gdstk not installed — skipping GDS export")

    print(cmp_path.as_posix())
    print(pol_img_path.as_posix())
    print(svg_path.as_posix())
    print(png_vec_path.as_posix())
    print(npz_out_path.as_posix())


def _save_smooth_gds(
    contours: List[np.ndarray],
    grid_shape: Tuple[int, int],
    design_size_um: float,
    gds_path: Path,
    layer: int,
    datatype: int,
) -> None:
    """Save smoothed contours as GDS polygons (in um coordinates)."""
    nx, ny = grid_shape
    dx = design_size_um / float(nx)
    dy = design_size_um / float(ny)
    x0 = -0.5 * design_size_um
    y0 = -0.5 * design_size_um

    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("PBS_POLISHED")

    for c in contours:
        # convert pixel coords → physical coords (um)
        # contour (row, col) → GDS (x, y): row → x, col → y
        pts = np.column_stack([
            x0 + c[:, 0] * dx,
            y0 + c[:, 1] * dy,
        ])
        poly = gdstk.Polygon(pts, layer=layer, datatype=datatype)
        cell.add(poly)

    lib.write_gds(gds_path.as_posix())


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    polish_pbs_structure(
        npz_path="logs/sim-guided/pbs_tsr=100_class=0_eta=1/samples_1x64x64x1_bin.npz",
        out_dir="results/pbs/structure_polished",
        sample_index=0,
        threshold=0.5,
        design_size_um=3.0,
        # gentler morphology: no opening, only closing
        open_radius=0,
        close_radius=1,
        min_area_px=4,
        # Gaussian smoothing for raster
        gauss_sigma=0.3,
        # contour smoothing for vector output
        smoothing_factor=2.0,
        num_points=300,
        min_contour_len=10,
    )
