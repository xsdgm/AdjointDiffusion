import argparse
import json
import random
from pathlib import Path
import sys
from typing import List, Optional

import numpy as np
import meep as mp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def _evaluate(individual: np.ndarray, step: int) -> float:
    fom, _ = pbs_sim(
        individual.reshape(64, 64),
        t=step,
        exp_name="pbs_baseline_ga",
        prop_dir="pbs",
    )
    return float(fom)


def run_ga(
    out_dir: str,
    npz_path: Optional[str],
    sample_index: int,
    seed: int,
    population_size: int,
    generations: int,
    mutation_rate: float,
    crossover_rate: float,
    elite_size: int,
) -> dict:
    mp.verbosity(0)
    np.random.seed(seed)
    random.seed(seed)

    n = 64 * 64
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    init = _load_initial_design(npz_path=npz_path, sample_index=sample_index, n=n)

    # Initialize binary population with one seed individual from input design.
    pop = []
    pop.append((init > 0.5).astype(float))
    for _ in range(population_size - 1):
        pop.append(np.random.randint(0, 2, size=n).astype(float))

    best_ind = None
    best_fom = -np.inf
    history = []

    def tournament_select(scores: List[float], k: int = 3) -> int:
        candidates = np.random.choice(len(scores), size=k, replace=False)
        return int(max(candidates, key=lambda idx: scores[idx]))

    for gen in range(generations):
        scores = [_evaluate(ind, step=gen) for ind in pop]

        gen_best_idx = int(np.argmax(scores))
        gen_best_fom = float(scores[gen_best_idx])
        gen_mean_fom = float(np.mean(scores))

        if gen_best_fom > best_fom:
            best_fom = gen_best_fom
            best_ind = pop[gen_best_idx].copy()

        print(
            f"[GA] gen={gen:03d} best={gen_best_fom:.6f} "
            f"mean={gen_mean_fom:.6f} global_best={best_fom:.6f}"
        )

        history.append(
            {
                "generation": gen,
                "best_fom": gen_best_fom,
                "mean_fom": gen_mean_fom,
                "global_best_fom": float(best_fom),
            }
        )

        elite_indices = np.argsort(scores)[-elite_size:][::-1]
        new_pop = [pop[i].copy() for i in elite_indices]

        while len(new_pop) < population_size:
            p1 = pop[tournament_select(scores)]
            p2 = pop[tournament_select(scores)]

            c1 = p1.copy()
            c2 = p2.copy()

            if np.random.rand() < crossover_rate:
                point = np.random.randint(1, n - 1)
                c1[:point], c1[point:] = p1[:point], p2[point:]
                c2[:point], c2[point:] = p2[:point], p1[point:]

            m1 = np.random.rand(n) < mutation_rate
            m2 = np.random.rand(n) < mutation_rate
            c1[m1] = 1.0 - c1[m1]
            c2[m2] = 1.0 - c2[m2]

            new_pop.append(c1)
            if len(new_pop) < population_size:
                new_pop.append(c2)

        pop = new_pop

    if best_ind is None:
        raise RuntimeError("GA failed to produce a valid individual.")

    saved_npz = _save_structure(best_ind, out_path)

    result = {
        "algorithm": "GA",
        "seed": seed,
        "population_size": population_size,
        "generations": generations,
        "mutation_rate": mutation_rate,
        "crossover_rate": crossover_rate,
        "elite_size": elite_size,
        "best_fom": float(best_fom),
        "output_npz": str(saved_npz),
        "history": history,
    }

    (out_path / "history.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"best_fom": result["best_fom"], "output_npz": result["output_npz"]}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="PBS baseline optimization using Genetic Algorithm.")
    parser.add_argument(
        "--npz_path",
        type=str,
        default="logs/sim-guided/pbs_tsr=100_class=0_eta=1/samples_1x64x64x1_bin.npz",
        help="Optional initial structure .npz path.",
    )
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="results/pbs/baseline_ga")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--population_size",
        type=int,
        default=16,
        help="GA population size (recommended: 12-16).",
    )
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--mutation_rate", type=float, default=0.01)
    parser.add_argument("--crossover_rate", type=float, default=0.9)
    parser.add_argument("--elite_size", type=int, default=2)
    args = parser.parse_args()

    run_ga(
        out_dir=args.out_dir,
        npz_path=args.npz_path,
        sample_index=args.sample_index,
        seed=args.seed,
        population_size=args.population_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        elite_size=args.elite_size,
    )


if __name__ == "__main__":
    main()
