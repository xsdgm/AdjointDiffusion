"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch as th

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood

import meep.adjoint as mpa
import meep as mp
from autograd import numpy as npa

import matplotlib.pyplot as plt
import os
import wandb
import pickle

from scipy.ndimage import label, binary_dilation
from skimage.measure import euler_number
from skimage.measure import label as label_


from guided_diffusion.simulation import CIS_sim, waveguide_sim


# log handler
if os.path.exists('lists.pkl'):
    os.remove('lists.pkl')

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["figure.figsize"] = (3.5,3.5)


params = {
    'axes.labelsize': 12,       # label font size
    'axes.titlesize': 12,       # title font size
    'xtick.labelsize': 10,      # x-axis tick label font size
    'ytick.labelsize': 10,      # y-axis tick label font size 
    'xtick.direction': 'in',    # tick mark direction (in, out, inout)
    'ytick.direction': 'in',    # tick mark direction (in, out, inout)
    'lines.markersize': 3,      # marker size
    'axes.titlepad': 6,         # padding between title and plot
    'axes.labelpad': 4,         # padding between axis label and plot
    'font.size': 12,            # font size
    #'font.sans-serif': 'Arial',  # font setting
    'figure.dpi': 300,          # resolution; for vector graphics, dpi doesn't affect output quality
    'figure.autolayout': True,  # automatic layout (ensures all elements are inside the figure)
    'xtick.top': True,          # display x-axis ticks on top
    'ytick.right': True,        # display y-axis ticks on right side
    'xtick.major.size': 2,      # length of x-axis major ticks
    'ytick.major.size': 2,      # length of y-axis major ticks
}

plt.rcParams.update(params)


def load_lists(filename='lists.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return [], [], []


# Function to delete every island with size 1
def delete_islands_with_size_1(array):
    # Label the connected components
    labeled_array, num_features = label(array)
    
    # Identify the features (islands) with size 1
    islands_to_delete = [i for i in range(1, num_features + 1) if np.sum(labeled_array == i) == 1]
    
    # Delete these islands
    for island in islands_to_delete:
        array[labeled_array == island] = 0
    
    if array.ndim > 2:
        array = np.squeeze(array, axis=2)
    else:
        pass
    return array


# Function to find the minimum feature size and its location
def find_minimum_feature_size(array):
    # Label the connected components
    labeled_array, num_features = label(array)
    
    # Get the sizes of all features
    feature_sizes = [(i, np.sum(labeled_array == i)) for i in range(1, num_features + 1)]
    
    # Find the minimum feature size and its label
    min_feature = min(feature_sizes, key=lambda x: x[1]) if feature_sizes else (0, 0)
    
    return min_feature[1], min_feature[0], labeled_array


def highlight_minimum_island(array, min_label, labeled_array):
    # Find the coordinates of the minimum island
    coords = np.argwhere(labeled_array == min_label)
    if coords.size == 0:
        return array
    
    # Create an empty mask for the boundary
    boundary_mask = np.zeros_like(array)
    
    # Fill the mask with the minimum island
    boundary_mask[labeled_array == min_label] = 1
    
    # Dilate the mask and subtract the original to get the boundary
    dilated_mask = binary_dilation(boundary_mask)
    boundary = dilated_mask - boundary_mask
    
    # Create a plot
    fig, ax = plt.subplots()
    ax.imshow(1-array, cmap='gray')   # black: 1, white: 0
    ax.imshow(boundary, cmap='Reds', alpha=0.5)  # Overlay the red boundary with higher alpha
    
    # Adjust boundary line thickness and make it more vivid
    boundary_indices = np.argwhere(boundary)
    for y, x in boundary_indices:
        rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, edgecolor='red', facecolor='red', linewidth=1)
        ax.add_patch(rect)
    
    return plt

def extract_elements(lst):
    # Create a list to store results.
    result = []

    # Iterate over the list and extract every 5th element.
    for i in range(0, len(lst), 5):
        result.append(lst[i])

    return result


def simulation_name(sim_name):  # sim: function
    if sim_name == "CIS_sim":
        return CIS_sim
    elif sim_name == "waveguide_sim":
        return waveguide_sim
    else:
        raise ValueError(f"Unsupported simulation name '{sim_name}'. Supported names are 'CIS' and 'waveguide'.")


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """
    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()     # the model predicts x_0
    EPSILON = enum.auto()     # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = enum.auto()  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()   # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """
    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        my_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        
        xt_grad = 0.0
        eta = 0.0
        t_cur = t[0].item()
        
        if my_kwargs['sim_guided'] and ( t_cur % my_kwargs['inter_rate'] == 0 ) and ( t_cur > my_kwargs['stoptime'] * self.num_timesteps ):
            assert x.shape[0] == 1  # only support batch size 1 for now
            
            from time import time
            start = time()
            
            eta = my_kwargs['eta']
            
            if my_kwargs['guidance_type'] == 'dps':
                # Goal: Compute the gradient of fom w.r.t. xt
                # To this end, first we need to compute the Jacobian matrix of x0hat w.r.t. xt
                x0hat_test = False
                
                def x0hat_from_xt(
                    xt,
                    t=t,
                    model=model,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs
                ):
                    return 0.5 * self.p_mean_variance(
                        model,
                        xt,
                        t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        model_kwargs=model_kwargs,
                    )['pred_xstart'] + 0.5
                
                with th.enable_grad():
                    xt = x.detach().clone().requires_grad_(True)
                    if x0hat_test:
                        print("x0hat test mode...")
                        x0hat_jacobian_xt = th.zeros(4096, 4096).to(x.device)
                    else:
                        x0hat_jacobian_xt = th.autograd.functional.jacobian(x0hat_from_xt, xt)
                        
                x0hat_jacobian_xt = x0hat_jacobian_xt.reshape(int(x0hat_jacobian_xt.numel() ** 0.5), -1)
                    
                # Computing the gradient of fom w.r.t. x0hat
                if x0hat_test:
                    fom = 0.0
                    adjoint_gradient = th.zeros_like(out['pred_xstart']).reshape(-1).float().to(x.device)
                else:
                    fom, adjoint_gradient = my_kwargs['simulation_'](
                        out['pred_xstart'].detach().cpu().numpy() * 0.5 + 0.5,
                        t_cur,
                        my_kwargs['exp_name'],
                        my_kwargs['prop_dir'],
                        my_kwargs['save_inter'],
                        my_kwargs['interval'],
                    )
                    adjoint_gradient = th.from_numpy(adjoint_gradient.reshape(-1)).float().to(x.device)
                    adjgrad_norm = adjoint_gradient.view(x.shape[0], -1).norm(dim=-1).mean().item()
                
                # Now we can compute the gradient of fom w.r.t. xt
                xt_grad = th.matmul(adjoint_gradient, x0hat_jacobian_xt).reshape(x.shape)
                xtgrad_norm = xt_grad.view(x.shape[0], -1).norm(dim=-1).mean().item()
                
                if my_kwargs['use_normed_grad']:
                    max_grad_values = xt_grad.abs().amax(dim=(1,2,3)).view(-1, 1, 1, 1)
                    xt_grad = xt_grad / max_grad_values
                
                if my_kwargs['use_adjgrad_norm']:
                    eta = eta/adjgrad_norm 

                end = time()
                print('time elapsed:', end-start)
                print('eta:', eta)
                print(f'fom at step {wandb.config.tsr-t_cur}: ', fom)
                print('adjgrad_norm:', adjgrad_norm)
                print('xtgrad_norm:', xtgrad_norm)
                
                x_ = 1-x
                wandb.log({
                    'fom': fom,
                    'eta': eta,
                    'adjgrad_norm': adjgrad_norm,
                    'xtgrad_norm': xtgrad_norm,
                    "generated": [wandb.Image(th.squeeze(x_.detach().cpu()).numpy(), caption='step_'+str(wandb.config.tsr-t_cur)+'_fom_'+str(fom)[:5])]
                }, step=wandb.config.tsr-t_cur)

            elif my_kwargs['guidance_type'] == 'dds':
                fom, adjoint_gradient = my_kwargs['simulation_'](
                    out['pred_xstart'].detach().cpu().numpy() * 0.5 + 0.5,
                    t_cur,
                    my_kwargs['exp_name'],
                    my_kwargs['prop_dir'],
                    my_kwargs['save_inter'],
                    my_kwargs['interval'],
                )
                adjoint_gradient = th.from_numpy(adjoint_gradient.reshape(x.shape)).float().to(x.device)
                adjgrad_norm = adjoint_gradient.view(x.shape[0], -1).norm(dim=-1).mean().item()
                
                if my_kwargs['use_normed_grad']:
                    max_grad_values = adjoint_gradient.abs().amax(dim=(1,2,3)).view(-1, 1, 1, 1)
                    adjoint_gradient = adjoint_gradient / max_grad_values
                
                out["pred_xstart"] = (0.5 * out["pred_xstart"] + 0.5) + eta * adjoint_gradient
                out["pred_xstart"] = (2 * out["pred_xstart"] - 1).clamp(-1, 1)
                
                out["mean"], _, _ = self.q_posterior_mean_variance(
                    x_start=out["pred_xstart"], x_t=x, t=t
                )
                
                end = time()
                print('time elapsed:', end-start)
                if my_kwargs['use_adjgrad_norm']:
                    eta = eta/adjgrad_norm 
                print('eta:', eta)
                print(f'fom at step {wandb.config.tsr-t_cur}: ', fom)
                print('adjgrad_norm:', adjgrad_norm)
                
                x_ = 1-x
                wandb.log({
                    'fom': fom,
                    'eta': eta,
                    'adjgrad_norm': adjgrad_norm,
                    "generated": [wandb.Image(th.squeeze(x_.detach().cpu()).numpy(), caption='step_'+str(wandb.config.tsr-t_cur)+'_fom_'+str(fom)[:5])]
                }, step=wandb.config.tsr-t_cur)
                
            if t_cur % my_kwargs['interval'] == 0:
                hist_dir = os.path.join('figures', my_kwargs['exp_name'])
                os.makedirs(hist_dir, exist_ok=True)
                plt.figure()
                
                plt.hist(th.squeeze(xt.detach().cpu() * 0.5 + 0.5).numpy().reshape(-1), bins=100, density=True)
                wandb.log({"pixel_dist_plot_xt": wandb.Image(plt, caption='step_'+str(wandb.config.tsr-t_cur)+'_fom_'+str(fom)[:5])}, step=wandb.config.tsr-t_cur)
                plt.savefig(os.path.join(hist_dir, 'xt_pixel_dist_step_'+str(wandb.config.tsr-t_cur)+'.png'))
                plt.close()

                plt.figure()
                plt.hist(th.squeeze(out['pred_xstart'].detach().cpu() * 0.5 + 0.5).numpy().reshape(-1), bins=100, density=True)
                wandb.log({"pixel_dist_plot_x0": wandb.Image(plt, caption='step_'+str(wandb.config.tsr-t_cur)+'_fom_'+str(fom)[:5])}, step=wandb.config.tsr-t_cur)
                plt.savefig(os.path.join(hist_dir, 'x0_pixel_dist_step_'+str(wandb.config.tsr-t_cur)+'.png'))
                plt.close()
                    
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise + eta * xt_grad
        
        if t_cur == 0:
            sample_bin = sample.detach().clone().cpu().numpy() * 0.5 + 0.5
            sample_bin_ = sample_bin
            sample_bin_[sample_bin_ > 0.5] = 1.0
            sample_bin_[sample_bin_ <= 0.5] = 0.0
            fom, sensitivity = my_kwargs['simulation_'](
                sample_bin_,
                t_cur,
                my_kwargs['exp_name'],
                my_kwargs['prop_dir'],
                my_kwargs['save_inter'],
                my_kwargs['interval'],
                flag_last=True
            )
            x_image = sample_bin_.reshape(64, 64, 1)
            print('fom (final, after binarization): ', fom)

            fig = plt.figure(figsize=(20,20))
            plt.imshow(np.squeeze(np.abs(sensitivity.reshape(64, 64))))
            plt.xlabel("x")
            plt.ylabel("y")
            plt.savefig("sensitivity.png")
            cmap = plt.cm.get_cmap()
            colormapping = plt.cm.ScalarMappable(cmap=cmap)
            cbar = fig.colorbar(colormapping, ax=plt.gca())
            wandb.log({"sensitivity": plt})

            x_image_ = 1-x_image

            wandb.log({
                'fom (binarized)': fom,
                "generated_binarized": [wandb.Image(x_image_)],
            }, step=wandb.config.tsr-t_cur)
            
            island_deleted = delete_islands_with_size_1(x_image.copy())
            min_feature_size, min_feature_label, labeled_array = find_minimum_feature_size(island_deleted)
            
            fom_island, _ = my_kwargs['simulation_'](
                island_deleted.flatten(),
                t_cur,
                my_kwargs['exp_name'],
                my_kwargs['prop_dir'],
                my_kwargs['save_inter'],
                my_kwargs['interval'],
                flag_last=True
            )
            
            wandb.log({
                'fom (binarized, island deleted1)': fom_island,
                "generated, island deleted1": [wandb.Image(1-island_deleted)],
            })
            
            array_converted_processed = delete_islands_with_size_1(1-island_deleted.copy())
            array_original = 1 - array_converted_processed
            min_feature_size2, min_feature_label2, labeled_array2 = find_minimum_feature_size(array_converted_processed)
            plt2 = highlight_minimum_island(array_original, min_feature_label2, labeled_array2)
            euler_number_original = euler_number(array_original, connectivity=1)
            _, num_islands1 = label_(island_deleted, connectivity=1, return_num=True)
            num_holes1 = num_islands1 - euler_number_original

            euler_number_converted = euler_number(array_converted_processed, connectivity=1)
            _, num_islands2 = label_(array_converted_processed, connectivity=1, return_num=True)
            num_holes2 = num_islands2 - euler_number_converted

            fom_island2, _ = my_kwargs['simulation_'](
                array_original.flatten(),
                t_cur,
                my_kwargs['exp_name'],
                my_kwargs['prop_dir'],
                my_kwargs['save_inter'],
                my_kwargs['interval'],
                flag_last=True
            )

            wandb.log({
               "highlighted, island deleted2": [wandb.Image(plt2)],
            })
            
            _, min_feature_label3, labeled_array3 = find_minimum_feature_size(array_original)
            plt3 = highlight_minimum_island(array_original, min_feature_label3, labeled_array3)
            
            wandb.log({
                'fom (final, binarized and island-deleted)': fom_island2,
                "generated_fianl": [wandb.Image(1-array_original)],
                "highlighted, final": [wandb.Image(plt3)]
            })

            print("array 0,0: ", array_original[0,0], ' value')

            wandb.log({
                'mfs (original)': min_feature_size,
                'mfs (converted)': min_feature_size2,
                'number of islands (original)': num_islands1,
                'number of islands (converted)': num_islands2,
                'number of holes (original)' : num_holes1,
                'number of holes (converted)': num_holes2,
                'euler_number with 1 connectivity (original)': euler_number_original,
                'euler_number with 1 connectivity (converted)': euler_number_converted,
            })

            np.save('final.npy', array_converted_processed)
            print('Array saved as final.npy.')

            if my_kwargs['sim_type'] == "CIS_sim":
                red_list, blue_list, green_list = load_lists()
                fred = extract_elements(red_list)
                fgreen = extract_elements(green_list)
                fblue = extract_elements(blue_list)
                
                columns = ['red', 'green', 'blue']
                wandb.log({"Intensity": wandb.plot.line_series(
                        xs = range(len(fred)),
                        ys = [fred, fgreen, fblue],
                        keys = columns,
                        xname = "Step"
                    )})

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        my_kwargs=None
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            my_kwargs=my_kwargs
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        my_kwargs=None
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        if my_kwargs['sim_guided']:
            assert my_kwargs['sim_type'] in ['CIS', 'waveguide']
            simulationlabel = my_kwargs['sim_type'] + '_sim'
            my_kwargs['simulation_'] = simulation_name(simulationlabel)
            print(my_kwargs['simulation_'])
            wandb.init(project=simulationlabel+"_diffusion")
            wandb.run.name = "class"+str(my_kwargs['manual_class_id'])+"_eta"+str(my_kwargs['eta'])+"_tsr"+str(my_kwargs['tsr'])

        cfg = {
            'eta': my_kwargs['eta'],
            'tsr': len(indices),
            'class': model_kwargs['y'][0].item() if model_kwargs is not None else None,
        }
        wandb.config.update(cfg)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    my_kwargs=my_kwargs
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )
        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
