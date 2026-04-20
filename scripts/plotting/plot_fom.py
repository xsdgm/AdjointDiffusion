import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


plt.rcParams.update({"font.family": "serif", "font.size": 12})


def plot_fom(csv_file: Path, out_dir: Path, stem: str = "fom_progress") -> None:
    steps: list[float] = []
    foms: list[float] = []

    with csv_file.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            steps.append(float(row["step"]))
            foms.append(float(row["fom"]))

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        steps,
        foms,
        color="#1f77b4",
        linewidth=1.5,
        marker="o",
        markersize=3,
        label="Figure of Merit (FOM)",
    )
    ax.set_xlabel("Diffusion Step")
    ax.set_ylabel("Figure of Merit (FOM)")
    ax.set_title("Optimization Trajectory via Adjoint-Guided Diffusion", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Generated FOM plots in {out_dir}")


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot FOM progression from intermediate sampling logs.")
    parser.add_argument(
        "--csv_file",
        type=Path,
        default=Path("logs/sim-guided/pbs_tsr=100_class=0_eta=1/intermediate_steps.csv"),
        help="Path to the intermediate_steps.csv file.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("figures"),
        help="Directory to store generated figures.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default="fom_progress",
        help="Base filename for generated plot files.",
    )
    return parser


def main() -> None:
    args = create_argparser().parse_args()
    plot_fom(args.csv_file, args.out_dir, stem=args.stem)


if __name__ == "__main__":
    main()
