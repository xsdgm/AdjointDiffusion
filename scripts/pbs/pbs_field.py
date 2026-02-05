from pathlib import Path

import matplotlib.pyplot as plt
import meep as mp
import numpy as np


def run_pbs_field(npz_path: str, out_dir: str, sample_index: int = 0, pol: str = "TE"):
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

    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(), 1e-7))

    ez = sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Ez)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"pbs_field_{pol}.npy"
    png_path = out_dir / f"pbs_field_{pol}.png"
    np.save(npy_path, ez)

    plt.figure(figsize=(4.2, 4.2))
    plt.imshow(ez.T, cmap="RdBu", origin="lower")
    plt.colorbar(label="Ez")
    plt.title(f"PBS Field ({pol})")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()

    print(npy_path.as_posix())
    print(png_path.as_posix())


if __name__ == "__main__":
    run_pbs_field(
        npz_path="results/pbs/samples_1x64x64x1_bin.npz",
        out_dir="results/pbs/fields",
        sample_index=0,
        pol="TE",
    )
    run_pbs_field(
        npz_path="results/pbs/samples_1x64x64x1_bin.npz",
        out_dir="results/pbs/fields",
        sample_index=0,
        pol="TM",
    )
