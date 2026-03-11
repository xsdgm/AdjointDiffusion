"""
SAC Multi-Episode Pretraining Script for Diffusion Guidance.

Trains a SAC agent over multiple episodes to learn how to guide
diffusion model sampling for photonic design optimization.

Each episode runs a full diffusion sampling trajectory (100 steps),
collecting (state, adjoint_grad, action, reward, ...) transitions
and updating the SAC agent via experience replay.

Usage:
    python guided_diffusion/train_sac.py \
        --model_path /path/to/diffusion_model.pt \
        --num_episodes 100 \
        --sim_type pbs --prop_dir pbs
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.sac_agent import SACAgent
from guided_diffusion.sim_env import SimEnvWrapper
from guided_diffusion.simulation import CIS_sim, waveguide_sim, pbs_sim_wideband


def resolve_hf_checkpoint(model_path, logger, label):
    if not model_path or not model_path.startswith("hf:"):
        return model_path
    spec = model_path[3:]
    if "/" not in spec:
        raise ValueError("hf: path must be 'hf:repo_id/filename'")
    repo_id, filename = spec.rsplit("/", 1)
    from huggingface_hub import hf_hub_download
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
    logger.log(f"Resolved {label} from HF: {repo_id}/{filename} -> {local_path}")
    return local_path


def upload_to_hf(local_path, repo_id, commit_message, logger):
    try:
        from huggingface_hub import create_repo, upload_file
    except ImportError as exc:
        raise ImportError("huggingface_hub is required for HF uploads") from exc
    if not repo_id:
        return
    try:
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        upload_file(
            path_or_fileobj=local_path,
            path_in_repo=os.path.basename(local_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
        logger.log(f"Uploaded to HF: {repo_id}/{os.path.basename(local_path)}")
    except Exception as e:
        logger.log(f"[WARNING] HF upload failed (training continues): {e}")


def simulation_name(sim_type):
    """Get simulation function by name. SAC always uses wideband for PBS."""
    sim_map = {
        'CIS_sim': CIS_sim,
        'waveguide_sim': waveguide_sim,
        'pbs_sim': pbs_sim_wideband,  # SAC uses multi-wavelength
    }
    return sim_map[sim_type]


def run_episode(model, diffusion, sac_agent, sim_env, args, episode_id, device, log_file):
    """
    Run one episode of diffusion sampling with SAC guidance.

    Returns:
        final_fom: the final figure of merit
        total_reward: sum of all rewards in the episode
        episode_steps: number of steps taken
    """
    shape = (1, 1, args.image_size, args.image_size)

    # Generate random noise (different per episode)
    img = torch.randn(*shape, device=device)

    # Setup model kwargs for class conditioning
    model_kwargs = {}
    if args.class_cond:
        classes = torch.ones(1, dtype=torch.long, device=device) * int(args.manual_class_id)
        model_kwargs["y"] = classes

    # Get timestep indices
    indices = list(range(diffusion.num_timesteps))[::-1]
    num_timesteps = diffusion.num_timesteps

    total_reward = 0.0
    episode_steps = 0
    fom = 0.0
    pending_transition = None

    for i in indices:
        t = torch.tensor([i], device=device)
        t_cur = i

        with torch.no_grad():
            # Get p_mean_variance prediction
            out = diffusion.p_mean_variance(
                model, img, t,
                clip_denoised=True,
                model_kwargs=model_kwargs,
            )

        if t_cur > args.stoptime * num_timesteps and t_cur < args.sac_start_ratio * num_timesteps:
            start = time.time()

            # Current state
            state = out['pred_xstart'].detach().clone()
            state_01 = state * 0.5 + 0.5

            # Get FoM and adjoint gradient
            fom_before, adjoint_grad = sim_env.evaluate_with_gradient(
                state_01.cpu().numpy(), t_cur
            )
            adjoint_grad_tensor = torch.from_numpy(
                adjoint_grad.reshape(state.shape)
            ).float().to(device)

            # SAC selects action
            timestep_norm = t_cur / num_timesteps

            # Now that the true next state for the previous step is available,
            # finalize and store the deferred transition.
            if pending_transition is not None:
                sac_agent.store_transition(
                    pending_transition['state'],
                    pending_transition['adjoint_grad'],
                    pending_transition['timestep_norm'],
                    pending_transition['action'],
                    pending_transition['reward'],
                    state.detach().clone(),
                    adjoint_grad_tensor.detach().clone(),
                    timestep_norm,
                    pending_transition['done'],
                )
                sac_agent.train_step()

            action = sac_agent.select_action(
                state, adjoint_grad_tensor, timestep_norm,
                deterministic=False
            )

            # Apply adaptive action (RL + Blended Adjoint)
            new_pred = sac_agent.apply_action(state, action, adjoint_grad_tensor)

            # Evaluate FoM after action and get next adjoint gradient in a single simulation call
            # (previously two separate calls on the same input — one was wasted)
            new_pred_01 = new_pred.detach() * 0.5 + 0.5
            fom_after, next_adjoint_grad = sim_env.evaluate_with_gradient(
                new_pred_01.cpu().numpy(), t_cur
            )
            next_adjoint_grad_tensor = torch.from_numpy(
                next_adjoint_grad.reshape(state.shape)
            ).float().to(device)

            reward_raw = fom_after - fom_before
            fom = fom_after
            total_reward += reward_raw

            # Defer insertion until next loop iteration where the real next
            # state (at t-1) is known.
            next_timestep_norm = max((t_cur - 1) / num_timesteps, 0.0)
            # Mark done if the next timestep will no longer be SAC-controlled
            done = (t_cur - 1) <= (args.stoptime * num_timesteps)
            pending_transition = {
                'state': state.detach().clone(),
                'adjoint_grad': adjoint_grad_tensor.detach().clone(),
                'timestep_norm': timestep_norm,
                'action': action.detach().clone(),
                'reward': reward_raw,
                'done': done,
                'fallback_next_state': new_pred.detach().clone(),
                'fallback_next_adjoint_grad': next_adjoint_grad_tensor.detach().clone(),
                'fallback_next_timestep_norm': next_timestep_norm,
            }

            # Update pred_xstart
            out['pred_xstart'] = new_pred.clamp(-1, 1)
            out['mean'], _, _ = diffusion.q_posterior_mean_variance(
                x_start=out['pred_xstart'], x_t=img, t=t
            )

            elapsed = time.time() - start
            episode_steps += 1

            # Per-step logging (every 10 steps)
            if episode_steps % 10 == 0:
                avg_critic, avg_actor = sac_agent.get_avg_loss()
                step_log = (f"  ep={episode_id:3d} step={episode_steps:4d} | t={t_cur:4d} | "
                           f"fom={fom:.6f} | reward={reward_raw:.6f} | "
                           f"alpha={sac_agent.alpha.item():.4f} | "
                           f"c_loss={avg_critic:.6f} | a_loss={avg_actor:.6f} | "
                           f"time={elapsed:.1f}s")
                print(step_log)
                if log_file:
                    log_file.write(step_log + '\n')
                    log_file.flush()

        # Sample x_{t-1}
        noise = torch.randn_like(img)
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)
        with torch.no_grad():
            img = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise

    # Final evaluation (binarized)
    if pending_transition is not None:
        # Terminal fallback: no further controlled timestep exists, so close
        # the transition with the last post-action state.
        sac_agent.store_transition(
            pending_transition['state'],
            pending_transition['adjoint_grad'],
            pending_transition['timestep_norm'],
            pending_transition['action'],
            pending_transition['reward'],
            pending_transition['fallback_next_state'],
            pending_transition['fallback_next_adjoint_grad'],
            pending_transition['fallback_next_timestep_norm'],
            True,
        )
        sac_agent.train_step()

    final_sample = img.detach().cpu().numpy() * 0.5 + 0.5
    final_sample[final_sample > 0.5] = 1.0
    final_sample[final_sample <= 0.5] = 0.0
    final_fom = sim_env.evaluate(final_sample, 0, flag_last=True)

    return final_fom, total_reward, episode_steps


def main():
    args = create_argparser().parse_args()

    if args.gpu_id != '':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    logger.configure(dir=log_dir)

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    args.model_path = resolve_hf_checkpoint(args.model_path, logger, "diffusion model")
    args.resume_path = resolve_hf_checkpoint(args.resume_path, logger, "SAC checkpoint")

    dist_util.setup_dist()
    device = dist_util.dev()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if world_size > 1 and rank != 0:
        logger.log(f"Rank {rank} is idle. SAC training and checkpointing run on rank 0 only.")
        return

    # Create model and diffusion
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # Create SAC agent
    sac_agent = SACAgent(
        image_size=args.image_size,
        patch_size=args.sac_patch_size,
        delta=args.sac_delta,
        lr=args.sac_lr,
        gamma=args.sac_gamma,
        tau=0.005,
        alpha_lr=args.sac_alpha_lr,
        buffer_size=args.sac_buffer_size,
        batch_size=args.sac_batch_size,
        reward_scale=args.sac_reward_scale,
        device=device,
    )

    # Load checkpoint if resuming
    if args.resume_path and os.path.exists(args.resume_path):
        sac_agent.load(args.resume_path)
        logger.log(f"Resumed SAC agent from {args.resume_path}")

    # Create simulation environment
    sim_label = args.sim_type + '_sim'
    sim_func = simulation_name(sim_label)
    sim_env = SimEnvWrapper(
        sim_func=sim_func,
        exp_name=f"sac_train_ep",
        prop_dir=args.prop_dir,
        save_inter=False,
        interval=1,
    )

    # Setup training log (ensure directory exists)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'sac_training_log.txt')
    with open(log_path, 'a') as log_file:
        log_file.write(f"\n=== SAC Pretraining Started ===\n")
        log_file.write(f"num_episodes: {args.num_episodes}\n")
        log_file.write(f"sim_type: {args.sim_type}\n")
        log_file.write(f"sac_lr: {args.sac_lr}\n")
        log_file.write(f"sac_delta: {args.sac_delta}\n")
        log_file.write(f"sac_patch_size: {args.sac_patch_size}\n")
        log_file.write(f"sac_batch_size: {args.sac_batch_size}\n")
        log_file.write(f"sac_buffer_size: {args.sac_buffer_size}\n")
        log_file.write(f"sac_gamma: {args.sac_gamma}\n")
        log_file.write(f"sac_reward_scale: {args.sac_reward_scale}\n")
        log_file.write(f"sac_start_ratio: {args.sac_start_ratio}\n")
        log_file.write(f"timestep_respacing: {args.timestep_respacing}\n")
        log_file.write(f"{'='*60}\n")
        log_file.flush()

        logger.log(f"Starting SAC pretraining for {args.num_episodes} episodes...")
        logger.log(f"Log file: {log_path}")

        best_fom = -float('inf')
        best_path = None

        for episode in range(args.num_episodes):
            ep_start = time.time()

            final_fom, total_reward, ep_steps = run_episode(
                model, diffusion, sac_agent, sim_env,
                args, episode, device, log_file
            )

            ep_elapsed = time.time() - ep_start

            # Episode summary
            ep_log = (f"EP {episode:3d}/{args.num_episodes} | "
                      f"final_fom={final_fom:.6f} | "
                      f"total_reward={total_reward:.6f} | "
                      f"steps={ep_steps} | "
                      f"buffer={len(sac_agent.replay_buffer)} | "
                      f"alpha={sac_agent.alpha.item():.4f} | "
                      f"time={ep_elapsed:.1f}s")
            print(f"\n{'='*70}")
            print(ep_log)
            print(f"{'='*70}\n")
            log_file.write(f"\n{ep_log}\n")
            log_file.flush()

            # Save checkpoint
            if (episode + 1) % args.save_interval == 0:
                ckpt_path = os.path.join(log_dir, f'sac_checkpoint_ep{episode+1}.pt')
                sac_agent.save(ckpt_path)
                logger.log(f"Checkpoint saved: {ckpt_path}")
                log_file.write(f"Checkpoint saved: {ckpt_path}\n")
                log_file.flush()
                # Upload checkpoint to HF
                if args.hf_repo_id and args.hf_upload_checkpoints:
                    upload_to_hf(ckpt_path, args.hf_repo_id, f"Checkpoint episode {episode+1}", logger)

            # Save best model
            if final_fom > best_fom:
                best_fom = final_fom
                best_path = os.path.join(log_dir, 'sac_best.pt')
                sac_agent.save(best_path)
                logger.log(f"New best FoM={best_fom:.6f}, saved to {best_path}")
                log_file.write(f"New best FoM={best_fom:.6f}, saved to {best_path}\n")
                log_file.flush()
                # Upload best to HF
                if args.hf_repo_id and args.hf_upload_best:
                    upload_to_hf(best_path, args.hf_repo_id, f"Best model FoM={best_fom:.6f}", logger)

        # Save final model
        final_path = os.path.join(log_dir, 'sac_final.pt')
        sac_agent.save(final_path)
        logger.log(f"Training complete. Final model: {final_path}")
        logger.log(f"Best FoM: {best_fom:.6f}")

        # Upload final model to HF
        if args.hf_repo_id and args.hf_upload_final:
            upload_to_hf(final_path, args.hf_repo_id, "Final model after training", logger)

        log_file.write(f"\n=== Training Complete ===\n")
        log_file.write(f"Best FoM: {best_fom:.6f}\n")
        log_file.write(f"Final model: {final_path}\n")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        log_dir="./logs/sac-train",
        gray_imgs=True,
        num_classes=3,
        manual_class_id="0",
        gpu_id="0",
        sim_type='pbs',
        prop_dir='pbs',
        stoptime=0.0,
        # SAC training params
        num_episodes=100,
        save_interval=10,
        resume_path='',
        sac_lr=3e-4,
        sac_alpha_lr=3e-4,
        sac_delta=0.1,
        sac_patch_size=8,
        sac_batch_size=64,
        sac_buffer_size=50000,
        sac_gamma=0.99,
        sac_reward_scale=1.0,
        sac_start_ratio=0.5,
        # Hugging Face upload
        hf_repo_id="",
        hf_upload_checkpoints=True,
        hf_upload_best=True,
        hf_upload_final=True,
        hf_endpoint="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
