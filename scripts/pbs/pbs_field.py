from pathlib import Path

import matplotlib.pyplot as plt
import meep as mp
import meep.adjoint as mpa
import numpy as np

from guided_diffusion.pbs_builder import (
    build_pbs_simulation,
    get_pbs_dominant_component,
)
from guided_diffusion.pbs_platform import get_pbs_platform_config


def run_pbs_field(
    npz_path: str,
    out_dir: str,
    sample_index: int = 0,
    pol: str = "TE",
    platform: str = "soi",
):
    mp.verbosity(0)
    cfg = get_pbs_platform_config(platform)

    npz = np.load(npz_path)
    struct = npz["arr_0"][sample_index, :, :, 0].astype("float32") / 255.0

    fcen = 1 / cfg.wavelength_um
    sim_data = build_pbs_simulation(mp, mpa, struct, pol, platform=cfg.name)
    sim = sim_data["sim"]

    dominant_component = get_pbs_dominant_component(mp, cfg, pol)
    if cfg.simulation_dim == 3:
        dft_components = [mp.Ex, mp.Ey, mp.Ez]
        dft_size = mp.Vector3(cfg.cell_size_x, cfg.cell_size_y, 0)
    else:
        dft_components = [dominant_component] if pol == "TE" else [mp.Ex, mp.Ey]
        dft_size = mp.Vector3(cfg.cell_size_x, cfg.cell_size_y, cfg.cell_size_z)
    dft_obj = sim.add_dft_fields(
        dft_components, fcen, 0, 1, center=mp.Vector3(), size=dft_size
    )

    decay_component = dominant_component
    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            50, decay_component, mp.Vector3(), 1e-7
        )
    )

    if cfg.simulation_dim == 3:
        ex = sim.get_dft_array(dft_obj, mp.Ex, 0)
        ey = sim.get_dft_array(dft_obj, mp.Ey, 0)
        ez = sim.get_dft_array(dft_obj, mp.Ez, 0)
        field = np.sqrt(np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2)
        field_label = "|E| @ z=0"
        cmap = "inferno"
    elif pol == "TE":
        field = np.abs(sim.get_dft_array(dft_obj, dominant_component, 0))
        field_label = f"|{cfg.dominant_field_te}|"
        cmap = "inferno"
    else:
        ex = sim.get_dft_array(dft_obj, mp.Ex, 0)
        ey = sim.get_dft_array(dft_obj, mp.Ey, 0)
        field = np.sqrt(np.abs(ex) ** 2 + np.abs(ey) ** 2)
        field_label = "|E|"
        cmap = "inferno"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"pbs_field_{pol}.npy"
    png_path = out_dir / f"pbs_field_{pol}.png"
    np.save(npy_path, field)

    plt.figure(figsize=(4.2, 4.2))
    plt.imshow(field.T, cmap=cmap, origin="lower")
    plt.colorbar(label=field_label)
    plt.title(f"PBS Field ({pol})")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()

    print(npy_path.as_posix())
    print(png_path.as_posix())


if __name__ == "__main__":
    run_pbs_field(
        npz_path="results/pbs/structure_polished/polished_structure.npz",
        out_dir="results/pbs/fields_polished",
        sample_index=0,
        pol="TE",
    )
    run_pbs_field(
        npz_path="results/pbs/structure_polished/polished_structure.npz",
        out_dir="results/pbs/fields_polished",
        sample_index=0,
        pol="TM",
    )
