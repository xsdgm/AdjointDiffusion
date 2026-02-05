import json
from pathlib import Path

import meep as mp
import numpy as np


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
        ),
        mp.Block(
            center=mp.Vector3(y=Sy / 4), material=Si, size=mp.Vector3(1, Sy / 2, 0)
        ),
        mp.Block(
            center=mp.Vector3(y=-Sy / 4), material=Si, size=mp.Vector3(1, Sy / 2, 0)
        ),
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
    top_flux = sim.add_flux(
        fcen,
        df,
        nf,
        mp.FluxRegion(center=mp.Vector3(0, 2.5, 0), size=mp.Vector3(x=2)),
    )
    bottom_flux = sim.add_flux(
        fcen,
        df,
        nf,
        mp.FluxRegion(center=mp.Vector3(0, -2.5, 0), size=mp.Vector3(x=2)),
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


def evaluate_pbs_metrics(
    npz_path: str,
    out_json: str,
    out_csv: str,
    sample_index: int = 0,
    lam_min: float = 1.50,
    lam_max: float = 1.60,
    nf: int = 11,
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

    t_top_te = _safe_div(top_te, src_te)
    t_bottom_te = _safe_div(bottom_te, src_te)

    t_top_tm = _safe_div(top_tm, src_tm)
    t_bottom_tm = _safe_div(bottom_tm, src_tm)

    il_top_te = -10 * np.log10(np.clip(t_top_te, 1e-12, None))
    il_bottom_tm = -10 * np.log10(np.clip(t_bottom_tm, 1e-12, None))

    xt_te = 10 * np.log10(np.clip(t_bottom_te / (t_top_te + 1e-12), 1e-12, None))
    xt_tm = 10 * np.log10(np.clip(t_top_tm / (t_bottom_tm + 1e-12), 1e-12, None))

    bw_mask = (il_top_te <= 3.0) & (il_bottom_tm <= 3.0) & (xt_te <= -10.0) & (xt_tm <= -10.0)
    if np.any(bw_mask):
        bw_lam = lam[bw_mask]
        bw = float(bw_lam.max() - bw_lam.min())
    else:
        bw = 0.0

    results = {
        "npz_path": str(npz_path),
        "sample_index": sample_index,
        "lambda_um": lam.tolist(),
        "T_top_TE": t_top_te.tolist(),
        "T_bottom_TE": t_bottom_te.tolist(),
        "T_top_TM": t_top_tm.tolist(),
        "T_bottom_TM": t_bottom_tm.tolist(),
        "IL_top_TE_dB": il_top_te.tolist(),
        "IL_bottom_TM_dB": il_bottom_tm.tolist(),
        "XT_TE_dB": xt_te.tolist(),
        "XT_TM_dB": xt_tm.tolist(),
        "bandwidth_um": bw,
        "bandwidth_criteria": {
            "IL_top_TE_dB<=3": True,
            "IL_bottom_TM_dB<=3": True,
            "XT_TE_dB<=-10": True,
            "XT_TM_dB<=-10": True,
        },
    }

    out_json = Path(out_json)
    out_csv = Path(out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(results, indent=2))

    header = [
        "lambda_um",
        "T_top_TE",
        "T_bottom_TE",
        "T_top_TM",
        "T_bottom_TM",
        "IL_top_TE_dB",
        "IL_bottom_TM_dB",
        "XT_TE_dB",
        "XT_TM_dB",
    ]
    data = np.column_stack(
        [
            lam,
            t_top_te,
            t_bottom_te,
            t_top_tm,
            t_bottom_tm,
            il_top_te,
            il_bottom_tm,
            xt_te,
            xt_tm,
        ]
    )
    np.savetxt(out_csv, data, delimiter=",", header=",".join(header), comments="")

    print(json.dumps({"bandwidth_um": bw, "out_json": str(out_json), "out_csv": str(out_csv)}, indent=2))


if __name__ == "__main__":
    evaluate_pbs_metrics(
        npz_path="logs/sim-guided/top_tsr=100_class=0_eta=1/samples_1x64x64x1.npz",
        out_json="logs/sim-guided/top_tsr=100_class=0_eta=1/pbs_metrics.json",
        out_csv="logs/sim-guided/top_tsr=100_class=0_eta=1/pbs_metrics.csv",
        sample_index=0,
        lam_min=1.50,
        lam_max=1.60,
        nf=11,
    )
