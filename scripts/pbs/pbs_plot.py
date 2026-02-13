from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_pbs_metrics(csv_path: str, out_dir: str) -> None:
    data = np.genfromtxt(csv_path, delimiter=",", names=True)

    lam = data["lambda_um"]
    t_front_te = data["T_front_TE"]
    t_front_tm = data["T_front_TM"]
    il_front_te = data["IL_front_TE_dB"]
    il_front_tm = data["IL_front_TM_dB"]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Transmission spectra
    plt.figure(figsize=(4.5, 3.2))
    plt.plot(lam, t_front_te, label="T_front_TE")
    plt.plot(lam, t_front_tm, label="T_front_TM")
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Transmission")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "pbs_transmission.png", dpi=300)
    plt.close()

    # Insertion loss
    plt.figure(figsize=(4.5, 3.2))
    plt.plot(lam, il_front_te, label="IL_front_TE (dB)")
    plt.plot(lam, il_front_tm, label="IL_front_TM (dB)")
    plt.axhline(3.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Insertion Loss (dB)")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "pbs_insertion_loss.png", dpi=300)
    plt.close()

    print(str(out_dir / "pbs_transmission.png"))
    print(str(out_dir / "pbs_insertion_loss.png"))


if __name__ == "__main__":
    plot_pbs_metrics(
        csv_path="logs/sim-guided/pbs_tsr=100_class=0_eta=1/pbs_metrics.csv",
        out_dir="logs/sim-guided/pbs_tsr=100_class=0_eta=1/figures",
    )
