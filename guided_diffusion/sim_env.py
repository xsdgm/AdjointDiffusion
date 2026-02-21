"""
Simulation Environment Wrapper for DQN-guided diffusion.

Wraps existing simulation functions (waveguide_sim, pbs_sim, CIS_sim) 
to provide a simple interface that only returns the figure of merit (fom)
without requiring adjoint gradients.
"""

import numpy as np


class SimEnvWrapper:
    """
    Wraps a simulation function to provide a DQN-compatible interface.
    
    The existing simulation functions return (fom, gradient). This wrapper
    calls the same function but only returns fom, discarding the gradient.
    """

    def __init__(self, sim_func, exp_name, prop_dir='top',
                 save_inter=False, interval=1):
        """
        Args:
            sim_func: the simulation function (e.g., waveguide_sim, pbs_sim)
            exp_name: experiment name for logging
            prop_dir: propagation direction
            save_inter: whether to save intermediate results
            interval: save interval
        """
        self.sim_func = sim_func
        self.exp_name = exp_name
        self.prop_dir = prop_dir
        self.save_inter = save_inter
        self.interval = interval

    def evaluate(self, struct_np, t=0, flag_last=False):
        """
        Run simulation and return only the fom.

        Args:
            struct_np: structure array (can be tensor or numpy)
            t: current diffusion timestep (for logging)
            flag_last: whether this is the final evaluation

        Returns:
            fom: float, figure of merit
        """
        # Convert to numpy if needed
        if hasattr(struct_np, 'detach'):
            struct_np = struct_np.detach().cpu().numpy()

        struct_np = np.squeeze(struct_np)
        struct_np = np.clip(struct_np, 0, 1)

        fom, _ = self.sim_func(
            struct_np,
            t,
            self.exp_name,
            self.prop_dir,
            self.save_inter,
            self.interval,
            flag_last=flag_last,
        )

        return float(fom)

    def evaluate_with_gradient(self, struct_np, t=0, flag_last=False):
        """
        Run simulation and return both fom and gradient.
        (Kept for compatibility, but DQN mode does not use gradients.)

        Args:
            struct_np: structure array
            t: current diffusion timestep
            flag_last: whether this is the final evaluation

        Returns:
            fom: float
            gradient: numpy array
        """
        if hasattr(struct_np, 'detach'):
            struct_np = struct_np.detach().cpu().numpy()

        struct_np = np.squeeze(struct_np)
        struct_np = np.clip(struct_np, 0, 1)

        fom, gradient = self.sim_func(
            struct_np,
            t,
            self.exp_name,
            self.prop_dir,
            self.save_inter,
            self.interval,
            flag_last=flag_last,
        )

        return float(fom), gradient
