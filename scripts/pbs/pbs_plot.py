from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_pbs_metrics(csv_path: str, out_dir: str) -> None:
    data = np.genfromtxt(csv_path, delimiter=",", names=True)

    lam = data["lambda_um"]
    t_top_te = data["T_top_TE"]
    t_bottom_te = data["T_bottom_TE"]
    t_top_tm = data["T_top_TM"]
    t_bottom_tm = data["T_bottom_TM"]
    il_top_te = data["IL_top_TE_dB"]
    il_bottom_tm = data["IL_bottom_TM_dB"]
    xt_te = data["XT_TE_dB"]
    xt_tm = data["XT_TM_dB"]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Transmission spectra
    plt.figure(figsize=(4.5, 3.2))
    plt.plot(lam, t_top_te, label="T_top_TE")
    plt.plot(lam, t_bottom_te, label="T_bottom_TE")
    plt.plot(lam, t_top_tm, label="T_top_TM")
    plt.plot(lam, t_bottom_tm, label="T_bottom_TM")
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Transmission")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "pbs_transmission.png", dpi=300)
    plt.close()

    # Insertion loss
    plt.figure(figsize=(4.5, 3.2))
    plt.plot(lam, il_top_te, label="IL_top_TE (dB)")
    plt.plot(lam, il_bottom_tm, label="IL_bottom_TM (dB)")
    plt.axhline(3.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Insertion Loss (dB)")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "pbs_insertion_loss.png", dpi=300)
    plt.close()

    # Crosstalk
    plt.figure(figsize=(4.5, 3.2))
    plt.plot(lam, xt_te, label="XT_TE (dB)")
    plt.plot(lam, xt_tm, label="XT_TM (dB)")
    plt.axhline(-10.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Crosstalk (dB)")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "pbs_crosstalk.png", dpi=300)
    plt.close()

    print(str(out_dir / "pbs_transmission.png"))
    print(str(out_dir / "pbs_insertion_loss.png"))
    print(str(out_dir / "pbs_crosstalk.png"))


if __name__ == "__main__":
    plot_pbs_metrics(
        csv_path="logs/sim-guided/top_tsr=100_class=0_eta=1/pbs_metrics.csv",
        out_dir="logs/sim-guided/top_tsr=100_class=0_eta=1/figures",
    )
