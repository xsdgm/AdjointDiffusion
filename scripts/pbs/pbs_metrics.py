import json
from pathlib import Path

import matplotlib.pyplot as plt
import meep as mp
import numpy as np


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _build_sim(struct, pol, fcen, fwidth, resolution=21):
    Si = mp.Medium(index=3.4)
    SiO2 = mp.Medium(index=1.44)

    Sx = 10
    Sy = 10
    cell_size = mp.Vector3(Sx, Sy)
    pml_layers = [mp.PML(2.0)]

    source_center = [-2.7, 0, 0]
    source_size = mp.Vector3(0, 2, 0)
    kpoint = mp.Vector3(1, 0, 0)

    Nx = 64
    Ny = 64

    # Output waveguide parameters
    y_offset = 0.8
    wg_width = 0.5

    parity = mp.ODD_Z if pol == "TE" else mp.EVEN_Z
    src = mp.GaussianSource(frequency=fcen, fwidth=fwidth)
    sources = [
        mp.EigenModeSource(
            src,
            eig_parity=parity,
            eig_band=1,
            direction=mp.NO_DIRECTION,
            eig_kpoint=kpoint,
            size=source_size,
            center=source_center,
        )
    ]

    design_variables = mp.MaterialGrid(
        mp.Vector3(Nx, Ny), SiO2, Si, grid_type="U_MEAN"
    )
    design_variables.update_weights(struct.flatten())
    design_region = mp.Volume(center=mp.Vector3(), size=mp.Vector3(3, 3, 0))

    geometry = [
        mp.Block(
            center=mp.Vector3(x=-Sx / 4), material=Si, size=mp.Vector3(Sx / 2, 1, 0)
        ), # Input waveguide
        mp.Block(
            center=mp.Vector3(x=Sx / 4, y=y_offset), material=Si, size=mp.Vector3(Sx / 2, wg_width, 0)
        ), # Output waveguide TE (top)
        mp.Block(
            center=mp.Vector3(x=Sx / 4, y=-y_offset), material=Si, size=mp.Vector3(Sx / 2, wg_width, 0)
        ), # Output waveguide TM (bottom)
        mp.Block(
            center=design_region.center, size=design_region.size, material=design_variables
        ),
    ]

    sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
        eps_averaging=True,
        subpixel_tol=1e-4,
        resolution=resolution,
    )

    return sim


def _run_flux(sim, fcen, df, nf):
    source_flux = sim.add_flux(
        fcen,
        df,
        nf,
        mp.FluxRegion(center=mp.Vector3(-2.5, 0, 0), size=mp.Vector3(y=2)),
    )
    # Output waveguide TE (top), y=0.8
    top_flux = sim.add_flux(
        fcen,
        df,
        nf,
        mp.FluxRegion(center=mp.Vector3(2.5, 0.8, 0), size=mp.Vector3(y=1.0)),
    )
    # Output waveguide TM (bottom), y=-0.8
    bottom_flux = sim.add_flux(
        fcen,
        df,
        nf,
        mp.FluxRegion(center=mp.Vector3(2.5, -0.8, 0), size=mp.Vector3(y=1.0)),
    )

    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(), 1e-7))

    src = np.array(mp.get_fluxes(source_flux))
    top = np.array(mp.get_fluxes(top_flux))
    bottom = np.array(mp.get_fluxes(bottom_flux))
    freqs = np.array(mp.get_flux_freqs(source_flux))

    sim.reset_meep()

    src = np.abs(src)
    top = np.abs(top)
    bottom = np.abs(bottom)

    return freqs, src, top, bottom


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pbs_metrics(csv_path: str, out_dir: str) -> None:
    """Read metrics CSV and generate transmission / IL / ER plots."""
    data = np.genfromtxt(csv_path, delimiter=",", names=True)

    lam = data["lambda_um"]
    t_front_te = data["T_front_TE"]
    t_front_tm = data["T_front_TM"]
    il_front_te = data["IL_front_TE_dB"]
    il_front_tm = data["IL_front_TM_dB"]
    er_te = data["ER_TE_dB"]
    er_tm = data["ER_TM_dB"]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 设置全局样式，符合学术论文风格
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'lines.linewidth': 1.5,
        'figure.figsize': (6, 4.5),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.format': 'png',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': ':',
    })

    # 定义颜色方案
    colors = {
        'te': '#1f77b4',  # 蓝色
        'tm': '#ff7f0e',  # 橙色
    }

    # Transmission spectra
    plt.figure()
    plt.plot(lam, t_front_te, label="TE", color=colors['te'], marker='o', markersize=4, markevery=5)
    plt.plot(lam, t_front_tm, label="TM", color=colors['tm'], marker='s', markersize=4, markevery=5)
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Transmission")
    plt.xlim(lam.min(), lam.max())
    plt.ylim(0, 1.05)
    plt.legend(loc='best', frameon=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_dir / "pbs_transmission.png")
    plt.close()

    # Insertion loss
    plt.figure()
    plt.plot(lam, il_front_te, label="TE", color=colors['te'], marker='o', markersize=4, markevery=5)
    plt.plot(lam, il_front_tm, label="TM", color=colors['tm'], marker='s', markersize=4, markevery=5)
    plt.axhline(3.0, color="gray", linestyle="--", linewidth=1, label="3 dB threshold")
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Insertion Loss (dB)")
    plt.xlim(lam.min(), lam.max())
    plt.ylim(0, max(il_front_te.max(), il_front_tm.max()) * 1.1)
    plt.legend(loc='best', frameon=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_dir / "pbs_insertion_loss.png")
    plt.close()

    # Extinction ratio
    plt.figure()
    plt.plot(lam, er_te, label="TE", color=colors['te'], marker='o', markersize=4, markevery=5)
    plt.plot(lam, er_tm, label="TM", color=colors['tm'], marker='s', markersize=4, markevery=5)
    plt.axhline(20.0, color="gray", linestyle="--", linewidth=1, label="20 dB threshold")
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Extinction Ratio (dB)")
    plt.xlim(lam.min(), lam.max())
    plt.ylim(0, max(er_te.max(), er_tm.max()) * 1.1)
    plt.legend(loc='best', frameon=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_dir / "pbs_extinction_ratio.png")
    plt.close()

    print(str(out_dir / "pbs_transmission.png"))
    print(str(out_dir / "pbs_insertion_loss.png"))
    print(str(out_dir / "pbs_extinction_ratio.png"))


# ---------------------------------------------------------------------------
# Metrics evaluation (runs simulation + saves data + plots)
# ---------------------------------------------------------------------------

def evaluate_pbs_metrics(
    npz_path: str,
    out_json: str,
    out_csv: str,
    out_fig_dir: str,
    sample_index: int = 0,
    lam_min: float = 1.50,
    lam_max: float = 1.60,
    nf: int = 41,
):
    mp.verbosity(0)

    npz = np.load(npz_path)
    struct = npz["arr_0"][sample_index, :, :, 0].astype("float32") / 255.0

    fmin = 1.0 / lam_max
    fmax = 1.0 / lam_min
    fcen = 0.5 * (fmin + fmax)
    df = fmax - fmin

    sim_te = _build_sim(struct, "TE", fcen, df)
    freqs, src_te, top_te, bottom_te = _run_flux(sim_te, fcen, df, nf)

    sim_tm = _build_sim(struct, "TM", fcen, df)
    _, src_tm, top_tm, bottom_tm = _run_flux(sim_tm, fcen, df, nf)

    lam = 1.0 / freqs

    def _safe_div(a, b):
        return a / (b + 1e-12)

    # For TE input: Signal is Top, Crosstalk is Bottom
    t_front_te = _safe_div(top_te, src_te)
    xt_front_te = _safe_div(bottom_te, src_te) # Crosstalk

    # For TM input: Signal is Bottom, Crosstalk is Top
    t_front_tm = _safe_div(bottom_tm, src_tm)
    xt_front_tm = _safe_div(top_tm, src_tm) # Crosstalk

    il_front_te = -10 * np.log10(np.clip(t_front_te, 1e-12, None))
    il_front_tm = -10 * np.log10(np.clip(t_front_tm, 1e-12, None))

    er_te = 10 * np.log10(np.clip( _safe_div(t_front_te, xt_front_te), 1e-12, None))
    er_tm = 10 * np.log10(np.clip( _safe_div(t_front_tm, xt_front_tm), 1e-12, None))

    min_er_te = float(np.min(er_te))
    min_er_tm = float(np.min(er_tm))

    bw_mask = (il_front_te <= 3.0) & (il_front_tm <= 3.0)
    if np.any(bw_mask):
        bw_lam = lam[bw_mask]
        bw = float(bw_lam.max() - bw_lam.min())
    else:
        bw = 0.0

    results = {
        "npz_path": str(npz_path),
        "sample_index": sample_index,
        "lambda_um": lam.tolist(),
        "T_front_TE": t_front_te.tolist(),
        "T_front_TM": t_front_tm.tolist(),
        "IL_front_TE_dB": il_front_te.tolist(),
        "IL_front_TM_dB": il_front_tm.tolist(),
        "ER_TE_dB": er_te.tolist(),
        "ER_TM_dB": er_tm.tolist(),
        "min_ER_TE_dB": min_er_te,
        "min_ER_TM_dB": min_er_tm,
        "bandwidth_um": bw,
        "bandwidth_criteria": {
            "IL_front_TE_dB<=3": True,
            "IL_front_TM_dB<=3": True,
        },
    }

    out_json = Path(out_json)
    out_csv = Path(out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(results, indent=2))

    header = [
        "lambda_um",
        "T_front_TE",
        "T_front_TM",
        "IL_front_TE_dB",
        "IL_front_TM_dB",
        "ER_TE_dB",
        "ER_TM_dB",
    ]
    data = np.column_stack(
        [
            lam,
            t_front_te,
            t_front_tm,
            il_front_te,
            il_front_tm,
            er_te,
            er_tm,
        ]
    )
    np.savetxt(out_csv, data, delimiter=",", header=",".join(header), comments="")

    print(json.dumps({
        "bandwidth_um": bw,
        "min_ER_TE_dB": min_er_te,
        "min_ER_TM_dB": min_er_tm,
        "out_json": str(out_json),
        "out_csv": str(out_csv)
    }, indent=2))

    # --- 仿真完成后自动绘图 ---
    plot_pbs_metrics(csv_path=str(out_csv), out_dir=out_fig_dir)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- 1. 原始结构仿真 --- 
    evaluate_pbs_metrics(
        npz_path="logs/sim-guided/pbs_tsr=100_class=0_eta=1/samples_1x64x64x1_bin.npz",
        out_json="results/pbs/pbs_metrics_raw.json",
        out_csv="results/pbs/pbs_metrics_raw.csv",
        out_fig_dir="results/pbs/figures_raw",
        sample_index=0,
        lam_min=1.50,
        lam_max=1.60,
        nf=41,
    )
    
    # --- 2. Polished 之后的仿真 --- 
    evaluate_pbs_metrics(
        npz_path="results/pbs/structure_polished/polished_structure.npz",
        out_json="results/pbs/pbs_metrics_polished.json",
        out_csv="results/pbs/pbs_metrics_polished.csv",
        out_fig_dir="results/pbs/figures_polished",
        sample_index=0,
        lam_min=1.50,
        lam_max=1.60,
        nf=41,
    )
    
    # --- 3. Shapeopt 之后的仿真 --- 
    evaluate_pbs_metrics(
        npz_path="results/pbs/shape_opt/optimized_structure.npz",
        out_json="results/pbs/pbs_metrics_shapeopt.json",
        out_csv="results/pbs/pbs_metrics_shapeopt.csv",
        out_fig_dir="results/pbs/figures_shapeopt",
        sample_index=0,
        lam_min=1.50,
        lam_max=1.60,
        nf=41,
    )
