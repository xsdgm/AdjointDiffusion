"""
PBS Shape Re-Optimization (Level Set + Adjoint)

Take the polished PBS structure as starting point and perform
boundary-only optimization using level set representation and
adjoint gradients from meep.adjoint.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
import meep as mp
import meep.adjoint as mpa


# ---------------------------------------------------------------------------
# Level Set utilities
# ---------------------------------------------------------------------------

def struct_to_levelset(binary: np.ndarray) -> np.ndarray:
    """Convert binary structure to Signed Distance Function (SDF).

    φ > 0 inside Si, φ < 0 inside SiO2.
    """
    binary = binary > 0.5
    dist_inside = distance_transform_edt(binary)
    dist_outside = distance_transform_edt(~binary)
    return dist_inside - dist_outside


def levelset_to_weights(phi: np.ndarray, beta: float = 8.0) -> np.ndarray:
    """Project level set to continuous weights [0, 1] via sigmoid."""
    return 1.0 / (1.0 + np.exp(-beta * phi))


def compute_boundary_mask(phi: np.ndarray, band_width: float = 3.0) -> np.ndarray:
    """Mask: 1.0 at boundary (|φ| < band_width), 0.0 elsewhere."""
    return (np.abs(phi) < band_width).astype(float)


# ---------------------------------------------------------------------------
# PBS adjoint simulation (adapted from pbs_sim in simulation.py)
# ---------------------------------------------------------------------------

def _run_pbs_adjoint(struct: np.ndarray, cross_weight: float = 0.5):
    """
    Run PBS adjoint simulation for both TE and TM.
    Returns total FoM and gradient w.r.t. flattened design weights.
    """
    import autograd.numpy as npa

    Si = mp.Medium(index=3.4)
    SiO2 = mp.Medium(index=1.44)

    resolution = 21
    Sx, Sy = 10, 10
    cell_size = mp.Vector3(Sx, Sy)
    pml_layers = [mp.PML(2.0)]

    fcen = 1 / 1.55
    width = 0.2
    fwidth = width * fcen

    source_center = [-2.7, 0, 0]
    source_size = mp.Vector3(0, 2, 0)
    kpoint = mp.Vector3(1, 0, 0)

    Nx, Ny = 64, 64
    y_offset = 0.8
    wg_width = 0.5

    total_fom = 0.0
    total_grad = np.zeros(Nx * Ny)

    for pol in ["TE", "TM"]:
        try:
            mp.reset_meep()
        except Exception:
            pass

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
        design_region = mpa.DesignRegion(
            design_variables,
            volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(3, 3, 0)),
        )

        geometry = [
            mp.Block(
                center=mp.Vector3(x=-Sx / 4), material=Si,
                size=mp.Vector3(Sx / 2, 1, 0),
            ),
            mp.Block(
                center=mp.Vector3(x=Sx / 4, y=y_offset), material=Si,
                size=mp.Vector3(Sx / 2, wg_width, 0),
            ),
            mp.Block(
                center=mp.Vector3(x=Sx / 4, y=-y_offset), material=Si,
                size=mp.Vector3(Sx / 2, wg_width, 0),
            ),
            mp.Block(
                center=design_region.center, size=design_region.size,
                material=design_variables,
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

        port_source = mpa.EigenmodeCoefficient(
            sim,
            mp.Volume(center=mp.Vector3(-2.5, 0, 0), size=mp.Vector3(y=2)),
            mode=1, eig_parity=parity,
        )
        port_top = mpa.EigenmodeCoefficient(
            sim,
            mp.Volume(center=mp.Vector3(2.5, y_offset, 0), size=mp.Vector3(y=1.0)),
            mode=1, eig_parity=parity,
        )
        port_bottom = mpa.EigenmodeCoefficient(
            sim,
            mp.Volume(center=mp.Vector3(2.5, -y_offset, 0), size=mp.Vector3(y=1.0)),
            mode=1, eig_parity=parity,
        )

        desired_port = port_top if pol == "TE" else port_bottom
        cross_port = port_bottom if pol == "TE" else port_top
        ob_list = [port_source, desired_port, cross_port]

        def J(source_coef, desired_coef, cross_coef):
            denom = source_coef + 1e-12
            desired = npa.abs(desired_coef / denom) ** 2
            cross = npa.abs(cross_coef / denom) ** 2
            return desired - cross_weight * cross

        opt = mpa.OptimizationProblem(
            simulation=sim,
            objective_functions=J,
            objective_arguments=ob_list,
            design_regions=[design_region],
            fcen=fcen, df=0, nf=1,
        )

        flat = struct.flatten()
        opt.update_design([flat])
        fom, g = opt([flat])

        total_fom += fom[0]
        total_grad += g.flatten()

    return total_fom, total_grad


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------

def run_shape_optimization(
    npz_path: str,
    out_dir: str,
    sample_index: int = 0,
    n_iters: int = 40,
    lr: float = 0.08,
    band_width: float = 3.0,
    beta_init: float = 2.0,
    beta_final: float = 16.0,
    cross_weight: float = 0.5,
):
    mp.verbosity(0)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load polished structure
    npz = np.load(npz_path)
    struct = npz["arr_0"][sample_index, :, :, 0].astype("float32") / 255.0
    binary_init = (struct > 0.5).astype(float)

    # initialize level set
    phi = struct_to_levelset(binary_init).astype(float)

    # logging
    fom_history = []
    best_fom = -np.inf
    best_weights = None

    print(f"Starting shape optimization: {n_iters} iterations, lr={lr}")
    print(f"Band width={band_width}, beta: {beta_init} → {beta_final}")
    print(f"Cross weight={cross_weight}")

    for it in range(n_iters):
        # annealing beta
        t = it / max(n_iters - 1, 1)
        beta = beta_init + (beta_final - beta_init) * t

        # level set → continuous weights
        weights = levelset_to_weights(phi, beta=beta)

        # boundary mask
        mask = compute_boundary_mask(phi, band_width=band_width)

        # adjoint simulation
        fom, grad = _run_pbs_adjoint(weights, cross_weight=cross_weight)

        fom_history.append(float(fom))

        # track best
        binariness = float(np.mean((weights > 0.99) | (weights < 0.01)))
        if fom > best_fom:
            best_fom = fom
            best_weights = weights.copy()

        print(f"  iter {it:3d}/{n_iters}  FoM={fom:.4f}  "
              f"best={best_fom:.4f}  β={beta:.1f}  "
              f"binary={binariness:.1%}")

        # gradient ascent (maximize FoM)
        # chain rule: ∂FoM/∂φ = ∂FoM/∂w · ∂w/∂φ
        # ∂w/∂φ = β · σ(β·φ) · (1 - σ(β·φ))
        sigmoid_deriv = beta * weights * (1.0 - weights)
        grad_phi = grad.reshape(phi.shape) * sigmoid_deriv

        # apply boundary mask
        grad_phi *= mask

        # normalize gradient
        grad_norm = np.max(np.abs(grad_phi)) + 1e-12
        grad_phi /= grad_norm

        # update level set
        phi += lr * grad_phi

    # use best weights
    final_weights = best_weights if best_weights is not None else weights
    final_binary = (final_weights > 0.5).astype(np.uint8) * 255

    # save optimized structure
    optimized_npz_path = out_dir / "optimized_structure.npz"
    arr = final_binary[np.newaxis, :, :, np.newaxis]
    np.savez(optimized_npz_path, arr)
    print(f"\n{optimized_npz_path}")

    # save FoM history
    history_path = out_dir / "fom_history.json"
    history_path.write_text(json.dumps({
        "fom_history": fom_history,
        "best_fom": best_fom,
        "n_iters": n_iters,
        "lr": lr,
        "beta_init": beta_init,
        "beta_final": beta_final,
    }, indent=2))
    print(history_path)

    # plot convergence
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(range(len(fom_history)), fom_history, "o-", markersize=3, color="#1f77b4")
    ax.axhline(best_fom, color="green", linestyle="--", linewidth=0.8,
               label=f"best = {best_fom:.4f}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("FoM (TE+TM)")
    ax.set_title("Shape Optimization Convergence")
    ax.legend(frameon=False)
    plt.tight_layout()
    conv_path = out_dir / "convergence.png"
    plt.savefig(conv_path, dpi=300)
    plt.close()
    print(conv_path)

    # plot comparison: before vs after
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(binary_init.T, origin="lower", cmap="gray")
    axes[0].set_title(f"Polished (FoM≈{fom_history[0]:.3f})")

    axes[1].imshow((final_weights > 0.5).astype(float).T, origin="lower", cmap="gray")
    axes[1].set_title(f"Re-optimized (FoM={best_fom:.3f})")

    diff = (final_weights > 0.5).astype(float) - binary_init
    axes[2].imshow(diff.T, origin="lower", cmap="RdBu", vmin=-1, vmax=1)
    axes[2].set_title("Difference (blue=removed, red=added)")

    for ax in axes:
        ax.set_aspect("equal")
    plt.tight_layout()
    comp_path = out_dir / "comparison.png"
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(comp_path)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_shape_optimization(
        npz_path="results/pbs/structure_polished/polished_structure.npz",
        out_dir="results/pbs/shape_opt",
        sample_index=0,
        n_iters=40,
        lr=0.08,
        band_width=3.0,
        beta_init=2.0,
        beta_final=16.0,
        cross_weight=0.5,
    )
