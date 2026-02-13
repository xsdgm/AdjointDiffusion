import json
from pathlib import Path

import autograd.numpy as npa
import meep as mp
import meep.adjoint as mpa
import numpy as np


def run_pbs_eval(npz_path: str, out_path: str, sample_index: int = 0) -> None:
    mp.verbosity(0)

    npz = np.load(npz_path)
    struct = npz["arr_0"][sample_index, :, :, 0].astype("float32") / 255.0

    Si = mp.Medium(index=3.4)
    SiO2 = mp.Medium(index=1.44)

    resolution = 21
    Sx = 10
    Sy = 10
    cell_size = mp.Vector3(Sx, Sy)
    pml_layers = [mp.PML(2.0)]

    fcen = 1 / 1.55
    width = 0.2
    fwidth = width * fcen

    source_center = [-2.7, 0, 0]
    source_size = mp.Vector3(0, 2, 0)
    kpoint = mp.Vector3(1, 0, 0)

    Nx = 64
    Ny = 64
    cross_weight = 0.5

    def _safe_reset():
        try:
            mp.reset_meep()
        except Exception:
            pass

    def _run_pol(pol: str):
        _safe_reset()
        
        y_offset = 0.8
        wg_width = 0.5
        
        parity = mp.ODD_Z if pol == "TE" else mp.EVEN_Z
        src = mp.GaussianSource(frequency=fcen, fwidth=fwidth)
        source = [
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
        design_region = mpa.DesignRegion(
            design_variables,
            volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(3, 3, 0)),
        )

        geometry = [
            mp.Block(
                center=mp.Vector3(x=-Sx / 4), material=Si, size=mp.Vector3(Sx / 2, 1, 0)
            ),
            mp.Block(
                center=mp.Vector3(x=Sx / 4, y=y_offset), material=Si, size=mp.Vector3(Sx / 2, wg_width, 0)
            ),
            mp.Block(
                center=mp.Vector3(x=Sx / 4, y=-y_offset), material=Si, size=mp.Vector3(Sx / 2, wg_width, 0)
            ),
            mp.Block(
                center=design_region.center, size=design_region.size, material=design_variables
            ),
        ]

        sim = mp.Simulation(
            cell_size=cell_size,
            boundary_layers=pml_layers,
            geometry=geometry,
            sources=source,
            eps_averaging=True,
            subpixel_tol=1e-4,
            resolution=resolution,
        )

        port_source = mpa.EigenmodeCoefficient(
            sim,
            mp.Volume(center=mp.Vector3(-2.5, 0, 0), size=mp.Vector3(y=2)),
            mode=1,
            eig_parity=parity,
        )
        port_top = mpa.EigenmodeCoefficient(
            sim,
            mp.Volume(center=mp.Vector3(2.5, y_offset, 0), size=mp.Vector3(y=1.0)),
            mode=1,
            eig_parity=parity,
        )
        port_bottom = mpa.EigenmodeCoefficient(
            sim,
            mp.Volume(center=mp.Vector3(2.5, -y_offset, 0), size=mp.Vector3(y=1.0)),
            mode=1,
            eig_parity=parity,
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

        flattened_array = struct.flatten()
        opt.update_design([flattened_array])
        fom, g = opt([flattened_array])
        fom_top = float(np.asarray(fom[0]).item())
        fom_bottom = float(np.asarray(fom[1]).item())
        return fom_top, fom_bottom, g

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
        "sample_index": sample_index,
        "fom_te_top": float(fom_te_top),
        "fom_te_bottom": float(fom_te_bottom),
        "fom_tm_top": float(fom_tm_top),
        "fom_tm_bottom": float(fom_tm_bottom),
        "fom_pbs": float(fom_pbs),
        "cross_weight": float(cross_weight),
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
    run_pbs_eval(
        npz_path="logs/sim-guided/pbs_tsr=100_class=0_eta=1/samples_1x64x64x1.npz",
        out_path="logs/sim-guided/pbs_tsr=100_class=0_eta=1/pbs_eval.json",
        sample_index=0,
    )
