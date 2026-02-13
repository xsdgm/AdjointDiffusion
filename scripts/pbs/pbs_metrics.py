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


def _run_flux(sim, fcen, df, nf, pol):
    target_y = 0.8 if pol == "TE" else -0.8
    
    source_flux = sim.add_flux(
        fcen,
        df,
        nf,
        mp.FluxRegion(center=mp.Vector3(-2.5, 0, 0), size=mp.Vector3(y=2)),
    )
    front_flux = sim.add_flux(
        fcen,
        df,
        nf,
        mp.FluxRegion(center=mp.Vector3(2.5, target_y, 0), size=mp.Vector3(y=1.0)),
    )

    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(), 1e-7))

    src = np.array(mp.get_fluxes(source_flux))
    front = np.array(mp.get_fluxes(front_flux))
    freqs = np.array(mp.get_flux_freqs(source_flux))

    sim.reset_meep()

    src = np.abs(src)
    front = np.abs(front)

    return freqs, src, front


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
    freqs, src_te, front_te = _run_flux(sim_te, fcen, df, nf, "TE")

    sim_tm = _build_sim(struct, "TM", fcen, df)
    _, src_tm, front_tm = _run_flux(sim_tm, fcen, df, nf, "TM")

    lam = 1.0 / freqs

    def _safe_div(a, b):
        return a / (b + 1e-12)

    t_front_te = _safe_div(front_te, src_te)
    t_front_tm = _safe_div(front_tm, src_tm)

    il_front_te = -10 * np.log10(np.clip(t_front_te, 1e-12, None))
    il_front_tm = -10 * np.log10(np.clip(t_front_tm, 1e-12, None))

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
    ]
    data = np.column_stack(
        [
            lam,
            t_front_te,
            t_front_tm,
            il_front_te,
            il_front_tm,
        ]
    )
    np.savetxt(out_csv, data, delimiter=",", header=",".join(header), comments="")

    print(json.dumps({"bandwidth_um": bw, "out_json": str(out_json), "out_csv": str(out_csv)}, indent=2))


if __name__ == "__main__":
    evaluate_pbs_metrics(
        npz_path="logs/sim-guided/pbs_tsr=100_class=0_eta=1/samples_1x64x64x1.npz",
        out_json="logs/sim-guided/pbs_tsr=100_class=0_eta=1/pbs_metrics.json",
        out_csv="logs/sim-guided/pbs_tsr=100_class=0_eta=1/pbs_metrics.csv",
        sample_index=0,
        lam_min=1.50,
        lam_max=1.60,
        nf=11,
    )
