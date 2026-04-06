import argparse
import importlib
import json
from pathlib import Path
from typing import Optional

import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import tensor_jacobian_product

from guided_diffusion.simulation import pbs_sim


def _load_initial_design(npz_path: Optional[str], sample_index: int, n: int) -> np.ndarray:
    if not npz_path:
        return 0.5 * np.ones(n, dtype=float)

    arr = np.load(npz_path)["arr_0"][sample_index, :, :, 0].astype("float32")
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr.flatten().astype(float)


def _save_structure(flat_design: np.ndarray, out_dir: Path, file_name: str = "optimized_structure.npz") -> Path:
    binary = (flat_design.reshape(64, 64) > 0.5).astype(np.uint8) * 255
    packed = binary[np.newaxis, :, :, np.newaxis]
    out_path = out_dir / file_name
    np.savez(out_path, packed)
    return out_path


def run_nlopt(
    out_dir: str,
    npz_path: Optional[str],
    sample_index: int,
    algorithm: str,
    maxeval_per_beta: int,
    num_betas: int,
    beta_init: float,
    beta_scale: float,
    eta: float,
    minimum_length: float,
) -> dict:
    mp.verbosity(0)

    try:
        nlopt = importlib.import_module("nlopt")
    except ModuleNotFoundError as exc:
        raise ImportError(
            "nlopt is required for pbs_baseline_nlopt.py. Install it via requirements.txt."
        ) from exc

    n = 64 * 64
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    x = _load_initial_design(npz_path=npz_path, sample_index=sample_index, n=n)

    algo_map = {
        "MMA": nlopt.LD_MMA,
        "SLSQP": nlopt.LD_SLSQP,
    }
    if algorithm not in algo_map:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Choose from {list(algo_map)}")

    design_region_width = 3.0
    design_region_height = 3.0
    design_region_resolution = 21

    def mapping(raw_x: np.ndarray, eta_val: float, beta_val: float) -> np.ndarray:
        filtered = mpa.conic_filter(
            raw_x,
            minimum_length,
            design_region_width,
            design_region_height,
            design_region_resolution,
        )
        projected = mpa.tanh_projection(filtered, beta_val, eta_val)
        return projected.flatten()

    history = []
    step_id = 0
    current_beta = beta_init

    def objective(raw_x: np.ndarray, grad: np.ndarray, beta_val: float) -> float:
        nonlocal step_id

        mapped = mapping(raw_x, eta, beta_val)
        fom, g_mapped = pbs_sim(
            mapped.reshape(64, 64),
            t=step_id,
            exp_name="pbs_baseline_nlopt",
            prop_dir="pbs",
        )
        fom = float(fom)
        g_mapped = np.asarray(g_mapped).flatten()

        if grad.size > 0:
            grad[:] = tensor_jacobian_product(mapping, 0)(raw_x, eta, beta_val, g_mapped)

        history.append({"step": step_id, "beta": float(beta_val), "fom": fom})
        print(f"[NLOPT] step={step_id:04d} beta={beta_val:.3f} fom={fom:.6f}")
        step_id += 1
        return fom

    for _ in range(num_betas):
        x = np.clip(x, 0.0, 1.0)
        solver = nlopt.opt(algo_map[algorithm], n)
        solver.set_lower_bounds(0.0)
        solver.set_upper_bounds(1.0)

        if np.isinf(current_beta):
            beta_for_step = current_beta
        else:
            beta_for_step = float(current_beta)

        solver.set_max_objective(lambda a, g: objective(a, g, beta_for_step))
        solver.set_maxeval(maxeval_per_beta)
        x = solver.optimize(x)

        if not np.isinf(current_beta):
            current_beta *= beta_scale

    x = np.clip(x, 0.0, 1.0)
    x_final = mapping(x, eta, np.inf)

    final_fom, _ = pbs_sim(
        x_final.reshape(64, 64),
        t=step_id,
        exp_name="pbs_baseline_nlopt",
        prop_dir="pbs",
        flag_last=True,
    )
    final_fom = float(final_fom)

    saved_npz = _save_structure(x_final, out_path)

    result = {
        "algorithm": f"NLOPT-{algorithm}",
        "maxeval_per_beta": maxeval_per_beta,
        "num_betas": num_betas,
        "beta_init": beta_init,
        "beta_scale": beta_scale,
        "eta": eta,
        "minimum_length": minimum_length,
        "final_fom": final_fom,
        "output_npz": str(saved_npz),
        "history": history,
    }
    (out_path / "history.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"final_fom": final_fom, "output_npz": str(saved_npz)}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="PBS baseline optimization using NLopt.")
    parser.add_argument(
        "--npz_path",
        type=str,
        default="logs/sim-guided/pbs_tsr=100_class=0_eta=1/samples_1x64x64x1_bin.npz",
        help="Optional initial structure .npz path.",
    )
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="results/pbs/baseline_nlopt")
    parser.add_argument("--algorithm", type=str, default="MMA", choices=["MMA", "SLSQP"])
    parser.add_argument("--maxeval_per_beta", type=int, default=5)
    parser.add_argument("--num_betas", type=int, default=6)
    parser.add_argument("--beta_init", type=float, default=2.0)
    parser.add_argument("--beta_scale", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--minimum_length", type=float, default=0.224)
    args = parser.parse_args()

    run_nlopt(
        out_dir=args.out_dir,
        npz_path=args.npz_path,
        sample_index=args.sample_index,
        algorithm=args.algorithm,
        maxeval_per_beta=args.maxeval_per_beta,
        num_betas=args.num_betas,
        beta_init=args.beta_init,
        beta_scale=args.beta_scale,
        eta=args.eta,
        minimum_length=args.minimum_length,
    )


if __name__ == "__main__":
    main()
