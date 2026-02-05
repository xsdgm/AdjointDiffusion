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

    def _run_pol(pol: str, prefer_dir: str):
        _safe_reset()
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
            mp.Volume(center=mp.Vector3(0, 2.5, 0), size=mp.Vector3(x=2)),
            mode=1,
            eig_parity=parity,
        )
        port_bottom = mpa.EigenmodeCoefficient(
            sim,
            mp.Volume(center=mp.Vector3(0, -2.5, 0), size=mp.Vector3(x=2)),
            mode=1,
            eig_parity=parity,
            forward=False,
        )

        ob_list = [port_source, port_top, port_bottom]

        def J(source_coef, top_coef, bottom_coef):
            denom = source_coef + 1e-12
            top_ratio = npa.abs(top_coef / denom) ** 2
            bottom_ratio = npa.abs(bottom_coef / denom) ** 2
            if prefer_dir == "top":
                return top_ratio - cross_weight * bottom_ratio
            return bottom_ratio - cross_weight * top_ratio

        opt = mpa.OptimizationProblem(
            simulation=sim,
            objective_functions=J,
            objective_arguments=ob_list,
            design_regions=[design_region],
            fcen=fcen,
            df=0,
            nf=1,
        )

        flattened_array = struct.flatten()
        opt.update_design([flattened_array])
        fom, g = opt([flattened_array])
        return float(fom[0]), g

    fom_te, g_te = _run_pol("TE", "top")
    fom_tm, g_tm = _run_pol("TM", "bottom")

    results = {
        "npz_path": str(npz_path),
        "sample_index": sample_index,
        "fom_te": float(fom_te),
        "fom_tm": float(fom_tm),
        "fom_total": float(fom_te + fom_tm),
        "grad_te_min": float(g_te.min()),
        "grad_te_max": float(g_te.max()),
        "grad_tm_min": float(g_tm.min()),
        "grad_tm_max": float(g_tm.max()),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run_pbs_eval(
        npz_path="logs/sim-guided/top_tsr=100_class=0_eta=1/samples_1x64x64x1.npz",
        out_path="logs/sim-guided/top_tsr=100_class=0_eta=1/pbs_eval.json",
        sample_index=0,
    )
