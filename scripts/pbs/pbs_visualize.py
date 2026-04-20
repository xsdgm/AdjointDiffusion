import argparse
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
    export_gds: bool = True,
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

    if export_gds:
        if gdstk is None:
            print("Skipping GDS export because gdstk is not installed.")
        else:
            _save_gds(
                binary=binary.astype(bool),
                gds_path=gds_path,
                design_size_um=design_size_um,
                layer=gds_layer,
                datatype=gds_datatype,
            )

    print(gray_path.as_posix())
    print(bin_path.as_posix())
    if export_gds and gdstk is not None:
        print(gds_path.as_posix())


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize PBS structures from a sampled NPZ file.")
    parser.add_argument("--npz_path", type=str, required=True, help="Path to the sampled NPZ file.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to store visualization outputs.")
    parser.add_argument("--sample_index", type=int, default=0, help="Sample index within the NPZ array.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used to binarize the grayscale structure.",
    )
    parser.add_argument(
        "--design_size_um",
        type=float,
        default=3.0,
        help="Physical design size in micrometers for GDS export.",
    )
    parser.add_argument("--gds_layer", type=int, default=1, help="GDS layer index.")
    parser.add_argument("--gds_datatype", type=int, default=0, help="GDS datatype.")
    parser.add_argument(
        "--no_gds",
        action="store_true",
        help="Skip GDS export and only generate raster visualizations.",
    )
    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    visualize_pbs_structure(
        npz_path=args.npz_path,
        out_dir=args.out_dir,
        sample_index=args.sample_index,
        threshold=args.threshold,
        design_size_um=args.design_size_um,
        gds_layer=args.gds_layer,
        gds_datatype=args.gds_datatype,
        export_gds=not args.no_gds,
    )
