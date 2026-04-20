"""
PBS Fabrication Tolerance Analysis

Simulate over-etching (erosion) and under-etching (dilation) effects
on the polished PBS structure, evaluating performance degradation
across a range of manufacturing errors.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.morphology import disk, binary_erosion, binary_dilation
from skimage.transform import resize

from guided_diffusion.pbs_platform import get_pbs_platform_config

# reuse simulation functions from pbs_metrics
from pbs_metrics import _build_sim, _run_flux

import meep as mp


# ---------------------------------------------------------------------------
# Core: apply fabrication error
# ---------------------------------------------------------------------------

def apply_fabrication_error(
    struct_64: np.ndarray,
    delta_nm: float,
    design_size_um: float = 3.0,
    hi_res: int = 640,
) -> np.ndarray:
    """
    Simulate fabrication error by eroding/dilating the binary structure.

    Parameters
    ----------
    struct_64 : (64, 64) float array (0~1 weights for MaterialGrid)
    delta_nm  : fabrication error in nm
                negative = over-etch (Si shrinks, erosion)
                positive = under-etch (Si expands, dilation)
    design_size_um : physical design region size in um
    hi_res : high-resolution grid size for sub-pixel morphology

    Returns
    -------
    perturbed : (64, 64) float array with continuous weights [0, 1].
                Edge pixels get fractional values reflecting the
                sub-pixel area change, allowing Meep's MaterialGrid
                to capture the fabrication perturbation accurately.
    """
    if delta_nm == 0:
        return struct_64.copy()

    binary = struct_64 > 0.5

    # upsample to high resolution (nearest-neighbor to keep edges sharp)
    binary_hi = resize(binary.astype(float), (hi_res, hi_res), order=0,
                       anti_aliasing=False, preserve_range=True) > 0.5

    # pixel size at high resolution
    pixel_nm = design_size_um * 1000.0 / hi_res  # nm per pixel
    radius_px = max(1, round(abs(delta_nm) / pixel_nm))
    selem = disk(radius_px)

    if delta_nm < 0:
        # over-etch: Si shrinks
        binary_hi = binary_erosion(binary_hi, selem)
    else:
        # under-etch: Si expands
        binary_hi = binary_dilation(binary_hi, selem)

    # downsample back to 64x64 using AREA AVERAGING (anti_aliasing=True)
    # This produces continuous weights [0, 1] at boundary pixels,
    # so Meep's MaterialGrid can resolve sub-pixel changes.
    result = resize(binary_hi.astype(float), struct_64.shape, order=1,
                    anti_aliasing=True)
    return np.clip(result, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_tolerance_sweep(
    npz_path: str,
    out_dir: str,
    sample_index: int = 0,
    deltas_nm: list = None,
    lam_min: float = 1.50,
    lam_max: float = 1.60,
    nf: int = 11,
    platform: str = "soi",
):
    if deltas_nm is None:
        deltas_nm = [-20, -15, -10, -5, 0, 5, 10, 15, 20]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_pbs_platform_config(platform)

    npz = np.load(npz_path)
    struct_orig = npz["arr_0"][sample_index, :, :, 0].astype("float32") / 255.0

    fmin = 1.0 / lam_max
    fmax = 1.0 / lam_min
    fcen = 0.5 * (fmin + fmax)
    df = fmax - fmin

    mp.verbosity(0)

    all_results = []

    for delta in deltas_nm:
        print(f"\n--- delta = {delta:+d} nm ---")

        struct = apply_fabrication_error(
            struct_orig,
            delta,
            design_size_um=cfg.design_region_size_um,
        )
        si_ratio = float(struct.sum()) / struct.size

        # run TE simulation
        sim_te = _build_sim(struct, "TE", fcen, df, platform=cfg.name)
        freqs, src_te, top_te, bottom_te = _run_flux(sim_te, fcen, df, nf, platform=cfg.name, pol="TE")

        # run TM simulation
        sim_tm = _build_sim(struct, "TM", fcen, df, platform=cfg.name)
        _, src_tm, top_tm, bottom_tm = _run_flux(sim_tm, fcen, df, nf, platform=cfg.name, pol="TM")

        lam = 1.0 / freqs

        def _safe_div(a, b):
            return a / (b + 1e-12)

        t_te = _safe_div(top_te, src_te)
        t_tm = _safe_div(bottom_tm, src_tm)

        xt_te = _safe_div(bottom_te, src_te)
        xt_tm = _safe_div(top_tm, src_tm)

        il_te = -10 * np.log10(np.clip(t_te, 1e-12, None))
        il_tm = -10 * np.log10(np.clip(t_tm, 1e-12, None))

        er_te = 10 * np.log10(np.clip(_safe_div(t_te, xt_te), 1e-12, None))
        er_tm = 10 * np.log10(np.clip(_safe_div(t_tm, xt_tm), 1e-12, None))

        # 3dB bandwidth
        bw_mask = (il_te <= 3.0) & (il_tm <= 3.0)
        if np.any(bw_mask):
            bw = float(lam[bw_mask].max() - lam[bw_mask].min())
        else:
            bw = 0.0

        result = {
            "delta_nm": delta,
            "platform": cfg.name,
            "simulation_dim": cfg.simulation_dim,
            "si_ratio": si_ratio,
            "peak_T_TE": float(np.max(t_te)),
            "peak_T_TM": float(np.max(t_tm)),
            "min_ER_TE_dB": float(np.min(er_te)),
            "min_ER_TM_dB": float(np.min(er_tm)),
            "avg_ER_TE_dB": float(np.mean(er_te)),
            "avg_ER_TM_dB": float(np.mean(er_tm)),
            "bandwidth_nm": bw * 1000,  # convert to nm
        }
        all_results.append(result)
        print(json.dumps(result, indent=2))

    # save summary CSV
    csv_path = out_dir / "tolerance_sweep.csv"
    header = list(all_results[0].keys())
    data = np.array([[r[k] for k in header] for r in all_results])
    np.savetxt(csv_path, data, delimiter=",", header=",".join(header), comments="")

    # save JSON for completeness
    json_path = out_dir / "tolerance_sweep.json"
    json_path.write_text(json.dumps(all_results, indent=2))

    print(f"\n{csv_path}")
    print(json_path)

    # plot results
    plot_tolerance(str(csv_path), str(out_dir))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tolerance(csv_path: str, out_dir: str):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    out_dir = Path(out_dir)

    deltas = data["delta_nm"]

    # --- 1. Peak Transmission vs delta ---
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(deltas, data["peak_T_TE"], "o-", label="TE", color="#1f77b4")
    ax.plot(deltas, data["peak_T_TM"], "s-", label="TM", color="#ff7f0e")
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Fabrication Error (nm)")
    ax.set_ylabel("Peak Transmission")
    ax.legend(frameon=False)
    ax.set_title("Transmission vs Fabrication Tolerance")
    plt.tight_layout()
    p = out_dir / "tolerance_transmission.png"
    plt.savefig(p, dpi=300)
    plt.close()
    print(p)

    # --- 2. Min ER vs delta ---
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(deltas, data["min_ER_TE_dB"], "o-", label="TE", color="#1f77b4")
    ax.plot(deltas, data["min_ER_TM_dB"], "s-", label="TM", color="#ff7f0e")
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.axhline(10, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label="10 dB ref")
    ax.set_xlabel("Fabrication Error (nm)")
    ax.set_ylabel("Min Extinction Ratio (dB)")
    ax.legend(frameon=False)
    ax.set_title("Min ER vs Fabrication Tolerance")
    plt.tight_layout()
    p = out_dir / "tolerance_min_er.png"
    plt.savefig(p, dpi=300)
    plt.close()
    print(p)

    # --- 3. Bandwidth vs delta ---
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(deltas, data["bandwidth_nm"], "o-", color="#2ca02c")
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Fabrication Error (nm)")
    ax.set_ylabel("3-dB Bandwidth (nm)")
    ax.set_title("Bandwidth vs Fabrication Tolerance")
    plt.tight_layout()
    p = out_dir / "tolerance_bandwidth.png"
    plt.savefig(p, dpi=300)
    plt.close()
    print(p)

    # --- 4. Si ratio vs delta (sanity check) ---
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(deltas, data["si_ratio"] * 100, "o-", color="#9467bd")
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Fabrication Error (nm)")
    ax.set_ylabel("Si Fill Ratio (%)")
    ax.set_title("Si Fill Ratio vs Fabrication Tolerance")
    plt.tight_layout()
    p = out_dir / "tolerance_si_ratio.png"
    plt.savefig(p, dpi=300)
    plt.close()
    print(p)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_tolerance_sweep(
        npz_path="results/pbs/structure_polished/polished_structure.npz",
        out_dir="results/pbs/tolerance",
        sample_index=0,
        deltas_nm=[-20, -15, -10, -5, 0, 5, 10, 15, 20],
        lam_min=1.50,
        lam_max=1.60,
        nf=11,
    )
