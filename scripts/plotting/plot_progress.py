import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)


def read_progress_csv(csv_file: Path) -> dict[str, list[float]]:
    data: dict[str, list[float]] = {}
    with csv_file.open("r", newline="") as handle:
        reader = csv.reader(handle)
        headers = next(reader)
        for header in headers:
            data[header] = []

        for row in reader:
            for idx, value in enumerate(row):
                data[headers[idx]].append(float(value))
    return data


def plot_training_progress(csv_file: Path, out_dir: Path) -> None:
    data = read_progress_csv(csv_file)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(data["step"], data["loss"], color="tab:blue", alpha=0.9, label="Total Loss")
    axes[0].set_xlabel("Training Steps")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Total Loss During Training")

    axes[1].plot(data["step"], data["mse"], color="tab:orange", alpha=0.9, label="MSE")
    axes[1].set_xlabel("Training Steps")
    axes[1].set_ylabel("Mean Squared Error (MSE)")
    axes[1].set_title("MSE Loss")

    axes[2].plot(data["step"], data["vb"], color="tab:green", alpha=0.9, label="Variational Bound (VB)")
    axes[2].set_xlabel("Training Steps")
    axes[2].set_ylabel("Variational Bound")
    axes[2].set_title("VB Loss")

    plt.tight_layout()
    plt.savefig(out_dir / "training_progress.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(out_dir / "training_progress.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    quartile_fig, quartile_ax = plt.subplots(figsize=(6, 4))
    quartile_ax.plot(data["step"], data["loss_q0"], alpha=0.7, label="q0")
    quartile_ax.plot(data["step"], data["loss_q1"], alpha=0.7, label="q1")
    quartile_ax.plot(data["step"], data["loss_q2"], alpha=0.7, label="q2")
    quartile_ax.plot(data["step"], data["loss_q3"], alpha=0.7, label="q3")
    quartile_ax.set_xlabel("Training Steps")
    quartile_ax.set_ylabel("Quartile Losses")
    quartile_ax.set_title("Loss Across Timestep Quartiles")
    quartile_ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "quartile_losses.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(out_dir / "quartile_losses.png", bbox_inches="tight", dpi=300)
    plt.close(quartile_fig)

    print(f"Generated training progress plots in {out_dir}")


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot training losses from a progress CSV.")
    parser.add_argument(
        "--csv_file",
        type=Path,
        default=Path("logs/train_logs/progress.csv"),
        help="Path to the training progress CSV.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("figures/train_progress"),
        help="Directory to store generated figures.",
    )
    return parser


def main() -> None:
    args = create_argparser().parse_args()
    plot_training_progress(args.csv_file, args.out_dir)


if __name__ == "__main__":
    main()
