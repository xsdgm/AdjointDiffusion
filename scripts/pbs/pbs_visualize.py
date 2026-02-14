from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import gdstk
except Exception:
    gdstk = None


def _normalize_structure(raw: np.ndarray) -> np.ndarray:
    struct = raw.astype("float32")
    if float(struct.max()) > 1.5:
        struct = struct / 255.0
    return np.clip(struct, 0.0, 1.0)


def _binary_to_gds_polygons(binary: np.ndarray, design_size_um: float, layer: int, datatype: int):
    nx, ny = binary.shape
    dx = design_size_um / float(nx)
    dy = design_size_um / float(ny)
    x0 = -0.5 * design_size_um
    y0 = -0.5 * design_size_um

    rectangles = []
    xs, ys = np.where(binary)
    for ix, iy in zip(xs, ys):
        x_min = x0 + ix * dx
        x_max = x_min + dx
        y_min = y0 + iy * dy
        y_max = y_min + dy
        rectangles.append(
            gdstk.rectangle((x_min, y_min), (x_max, y_max), layer=layer, datatype=datatype)
        )

    if not rectangles:
        return []

    merged = gdstk.boolean(rectangles, [], "or", layer=layer, datatype=datatype)
    return merged if merged is not None else []


def _save_gds(
    binary: np.ndarray,
    gds_path: Path,
    design_size_um: float,
    layer: int,
    datatype: int,
) -> None:
    if gdstk is None:
        raise ImportError(
            "gdstk is required for GDSII export. Please install it with: pip install gdstk"
        )

    polygons = _binary_to_gds_polygons(binary, design_size_um, layer, datatype)
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("PBS_STRUCTURE")
    for poly in polygons:
        cell.add(poly)
    lib.write_gds(gds_path.as_posix())


def visualize_pbs_structure(
    npz_path: str,
    out_dir: str,
    sample_index: int = 0,
    threshold: float = 0.5,
    design_size_um: float = 3.0,
    gds_layer: int = 1,
    gds_datatype: int = 0,
) -> None:
    npz = np.load(npz_path)
    raw = npz["arr_0"][sample_index, :, :, 0]
    struct = _normalize_structure(raw)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gray_path = out_dir / "pbs_structure_gray.png"
    bin_path = out_dir / "pbs_structure_bin.png"
    gds_path = out_dir / "pbs_structure.gds"

    plt.figure(figsize=(4.2, 4.2))
    plt.imshow(struct.T, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Density")
    plt.title("PBS Structure (Grayscale)")
    plt.tight_layout()
    plt.savefig(gray_path, dpi=300)
    plt.close()

    binary = (struct > threshold).astype("float32")
    plt.figure(figsize=(4.2, 4.2))
    plt.imshow(binary.T, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Binary")
    plt.title(f"PBS Structure (Binary, thr={threshold:.2f})")
    plt.tight_layout()
    plt.savefig(bin_path, dpi=300)
    plt.close()

    _save_gds(
        binary=binary.astype(bool),
        gds_path=gds_path,
        design_size_um=design_size_um,
        layer=gds_layer,
        datatype=gds_datatype,
    )

    print(gray_path.as_posix())
    print(bin_path.as_posix())
    print(gds_path.as_posix())


if __name__ == "__main__":
    visualize_pbs_structure(
        npz_path="logs/sim-guided/pbs_tsr=100_class=0_eta=1/samples_1x64x64x1_bin.npz",
        out_dir="results/pbs/structure",
        sample_index=0,
        threshold=0.5,
        design_size_um=3.0,
        gds_layer=1,
        gds_datatype=0,
    )
