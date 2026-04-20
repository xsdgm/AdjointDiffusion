import json
from pathlib import Path

import autograd.numpy as npa
import meep as mp
import meep.adjoint as mpa
import numpy as np

from guided_diffusion.pbs_builder import (
    build_pbs_ports,
    build_pbs_simulation,
    get_objective_wavelengths,
)
from guided_diffusion.pbs_platform import get_pbs_platform_config


def run_pbs_eval(
    npz_path: str,
    out_path: str,
    sample_index: int = 0,
    polished_npz_path: str | None = None,
    platform: str = "soi",
) -> None:
    mp.verbosity(0)
    cfg = get_pbs_platform_config(platform)

    if polished_npz_path is not None:
        # --- 使用平滑后的结构 ---
        npz = np.load(polished_npz_path)
        struct = npz["arr_0"][sample_index, :, :, 0].astype("float32") / 255.0
        struct_source = str(polished_npz_path)
    else:
        # --- 使用原始结构 ---
        npz = np.load(npz_path)
        struct = npz["arr_0"][sample_index, :, :, 0].astype("float32") / 255.0
        struct_source = str(npz_path)

    objective_wavelengths_um, objective_weights = get_objective_wavelengths(cfg)
    cross_weight = cfg.cross_weight

    def _safe_reset():
        try:
            mp.reset_meep()
        except Exception:
            pass

    def _run_pol(pol: str):
        fom_top_acc = 0.0
        fom_bottom_acc = 0.0
        grad_acc = None
        for wavelength_um, weight in zip(objective_wavelengths_um, objective_weights):
            _safe_reset()
            fcen = 1 / wavelength_um
            sim_data = build_pbs_simulation(mp, mpa, struct, pol, platform=cfg.name, fcen=fcen)
            sim = sim_data["sim"]
            design_region = sim_data["design_region"]
            port_source, port_top, port_bottom = build_pbs_ports(
                mp,
                mpa,
                sim,
                cfg,
                sim_data["parity"],
            )

            def J_top(source_coef, top_coef, bottom_coef):
                denom = source_coef + 1e-12
                return npa.abs(top_coef / denom) ** 2

            def J_bottom(source_coef, top_coef, bottom_coef):
                denom = source_coef + 1e-12
                return npa.abs(bottom_coef / denom) ** 2

            opt = mpa.OptimizationProblem(
                simulation=sim,
                objective_functions=[J_top, J_bottom],
                objective_arguments=[port_source, port_top, port_bottom],
                design_regions=[design_region],
                fcen=fcen,
                df=0,
                nf=1,
            )

            flattened_array = sim_data["flattened_weights"]
            opt.update_design([flattened_array])
            fom, g = opt([flattened_array])
            fom_top_acc += weight * float(np.asarray(fom[0]).item())
            fom_bottom_acc += weight * float(np.asarray(fom[1]).item())
            if grad_acc is None:
                grad_acc = [weight * np.asarray(gi) for gi in g]
            else:
                grad_acc = [acc + weight * np.asarray(gi) for acc, gi in zip(grad_acc, g)]
        return fom_top_acc, fom_bottom_acc, grad_acc

    fom_te_top, fom_te_bottom, g_te = _run_pol("TE")
    fom_tm_top, fom_tm_bottom, g_tm = _run_pol("TM")

    fom_pbs = (fom_te_top - cross_weight * fom_te_bottom) + (fom_tm_bottom - cross_weight * fom_tm_top)

    def _grad_stats(grad):
        try:
            return float(np.min(grad)), float(np.max(grad))
        except Exception:
            return None, None

    def _split_grad(grad, idx):
        if isinstance(grad, (list, tuple)) and len(grad) > idx:
            return grad[idx]
        return None

    te_top_min, te_top_max = _grad_stats(_split_grad(g_te, 0))
    te_bottom_min, te_bottom_max = _grad_stats(_split_grad(g_te, 1))
    tm_top_min, tm_top_max = _grad_stats(_split_grad(g_tm, 0))
    tm_bottom_min, tm_bottom_max = _grad_stats(_split_grad(g_tm, 1))

    results = {
        "npz_path": str(npz_path),
        "struct_source": struct_source,
        "polished": polished_npz_path is not None,
        "sample_index": sample_index,
        "fom_te_top": float(fom_te_top),
        "fom_te_bottom": float(fom_te_bottom),
        "fom_tm_top": float(fom_tm_top),
        "fom_tm_bottom": float(fom_tm_bottom),
        "fom_pbs": float(fom_pbs),
        "cross_weight": float(cross_weight),
        "platform": cfg.name,
        "simulation_dim": cfg.simulation_dim,
        "crystal_cut": cfg.crystal_cut,
        "optic_axis": cfg.optic_axis,
        "objective_wavelengths_um": list(objective_wavelengths_um),
        "objective_wavelength_weights": list(objective_weights),
        "grad_te_top_min": te_top_min,
        "grad_te_top_max": te_top_max,
        "grad_te_bottom_min": te_bottom_min,
        "grad_te_bottom_max": te_bottom_max,
        "grad_tm_top_min": tm_top_min,
        "grad_tm_top_max": tm_top_max,
        "grad_tm_bottom_min": tm_bottom_min,
        "grad_tm_bottom_max": tm_bottom_max,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    # --- 评估原始结构 ---
    run_pbs_eval(
        npz_path="logs/sim-guided/pbs_tsr=100_class=0_eta=1/samples_1x64x64x1_bin.npz",
        out_path="results/pbs/pbs_eval.json",
        sample_index=0,
    )

    # --- 评估平滑后的结构 ---
    run_pbs_eval(
        npz_path="logs/sim-guided/pbs_tsr=100_class=0_eta=1/samples_1x64x64x1_bin.npz",
        out_path="results/pbs/pbs_eval_polished.json",
        sample_index=0,
        polished_npz_path="results/pbs/structure_polished/polished_structure.npz",
    )
