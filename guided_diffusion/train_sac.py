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
import csv
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


def scheduled_value(start, end, progress, mode="linear"):
    """Interpolate a scalar hyperparameter according to the selected schedule."""
    p = float(np.clip(progress, 0.0, 1.0))
    if mode == "cosine":
        p = 0.5 - 0.5 * np.cos(np.pi * p)
    return float(start + (end - start) * p)


def run_episode(model, diffusion, sac_agent, sim_env, args, episode_id, device, log_file, phase_ratio):
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
    sac_steps = 0
    adjoint_steps = 0
    control_min = args.stoptime * num_timesteps
    control_max = args.sac_start_ratio * num_timesteps
    control_span = max(control_max - control_min, 1e-8)
    phase_switch_t = control_min + (1.0 - phase_ratio) * control_span
    adj_stats = {
        'calls': 0,
        'time_ms_sum': 0.0,
        'norm_sum': 0.0,
        'norm_sq_sum': 0.0,
        'max_abs': 0.0,
        'nonfinite_count': 0,
        'near_zero_count': 0,
        'delta_norm_sum': 0.0,
        'delta_norm_count': 0,
        'prev_norm': None,
    }

    def record_adjoint_stats(grad_array, elapsed_seconds):
        grad_np = np.asarray(grad_array, dtype=np.float32)
        finite_mask = np.isfinite(grad_np)
        if not finite_mask.all():
            adj_stats['nonfinite_count'] += 1
            grad_np = np.nan_to_num(grad_np, nan=0.0, posinf=0.0, neginf=0.0)

        grad_flat = grad_np.reshape(-1)
        if grad_flat.size == 0:
            norm = 0.0
            max_abs = 0.0
        else:
            norm = float(np.linalg.norm(grad_flat))
            max_abs = float(np.max(np.abs(grad_flat)))

        if norm < 1e-8:
            adj_stats['near_zero_count'] += 1

        prev_norm = adj_stats['prev_norm']
        if prev_norm is not None:
            adj_stats['delta_norm_sum'] += abs(norm - prev_norm)
            adj_stats['delta_norm_count'] += 1
        adj_stats['prev_norm'] = norm

        adj_stats['calls'] += 1
        adj_stats['time_ms_sum'] += max(float(elapsed_seconds), 0.0) * 1000.0
        adj_stats['norm_sum'] += norm
        adj_stats['norm_sq_sum'] += norm * norm
        if max_abs > adj_stats['max_abs']:
            adj_stats['max_abs'] = max_abs

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

        if t_cur > control_min and t_cur < control_max:
            start = time.time()
            use_sac_phase = t_cur > phase_switch_t
            phase_tag = "SAC" if use_sac_phase else "ADJ"

            # Current state
            state = out['pred_xstart'].detach().clone()
            state_01 = state * 0.5 + 0.5

            # Get FoM and adjoint gradient
            adj_start = time.time()
            fom_before, adjoint_grad = sim_env.evaluate_with_gradient(
                state_01.cpu().numpy(), t_cur
            )
            record_adjoint_stats(adjoint_grad, time.time() - adj_start)
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
                pending_transition = None

            if use_sac_phase:
                action = sac_agent.select_action(
                    state, adjoint_grad_tensor, timestep_norm,
                    deterministic=False
                )

                # SAC phase: apply adaptive blended action and defer transition write.
                new_pred = sac_agent.apply_action(state, action, adjoint_grad_tensor)
                sac_steps += 1
            else:
                # Adjoint-only phase: deterministic local refinement, SAC policy is bypassed.
                action = None
                adj_norm = adjoint_grad_tensor.abs().amax().clamp(min=1e-8)
                pure_adjoint_step = (adjoint_grad_tensor / adj_norm) * args.sac_delta
                new_pred = (state + pure_adjoint_step).clamp(-1, 1)
                adjoint_steps += 1

            # Evaluate FoM after action and get next adjoint gradient in a single simulation call
            # (previously two separate calls on the same input — one was wasted)
            new_pred_01 = new_pred.detach() * 0.5 + 0.5
            adj_start = time.time()
            fom_after, next_adjoint_grad = sim_env.evaluate_with_gradient(
                new_pred_01.cpu().numpy(), t_cur
            )
            record_adjoint_stats(next_adjoint_grad, time.time() - adj_start)
            next_adjoint_grad_tensor = torch.from_numpy(
                next_adjoint_grad.reshape(state.shape)
            ).float().to(device)

            reward_raw = fom_after - fom_before
            fom = fom_after
            total_reward += reward_raw

            if use_sac_phase:
                # Defer insertion until next loop iteration where the real next
                # state (at t-1) is known.
                next_timestep_norm = max((t_cur - 1) / num_timesteps, 0.0)
                # Mark done if the next timestep exits SAC phase.
                done = (t_cur - 1) <= phase_switch_t
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
                monitor = sac_agent.get_monitor_snapshot()
                monitor_extra = ""
                if monitor:
                    adj_calls = max(adj_stats['calls'], 1)
                    adj_norm_mean = adj_stats['norm_sum'] / adj_calls
                    adj_time_ms = adj_stats['time_ms_sum'] / adj_calls
                    monitor_extra = (
                        f" | q_gap={monitor.get('q_gap_abs_mean', 0.0):.4f}"
                        f" | td_abs={monitor.get('td_error_abs_mean', 0.0):.4f}"
                        f" | ent={monitor.get('entropy_estimate', 0.0):.4f}"
                        f" | g(a/c1/c2)=({monitor.get('actor_grad_norm', 0.0):.3f}/"
                        f"{monitor.get('critic1_grad_norm', 0.0):.3f}/"
                        f"{monitor.get('critic2_grad_norm', 0.0):.3f})"
                        f" | adj_n={adj_norm_mean:.3e}"
                        f" | adj_ms={adj_time_ms:.1f}"
                    )
                step_log = (f"  ep={episode_id:3d} step={episode_steps:4d} | t={t_cur:4d} | "
                           f"phase={phase_tag} | "
                           f"fom={fom:.6f} | reward={reward_raw:.6f} | "
                           f"alpha={sac_agent.alpha.item():.4f} | "
                           f"c_loss={avg_critic:.6f} | a_loss={avg_actor:.6f} | "
                           f"time={elapsed:.1f}s"
                           f"{monitor_extra}")
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

    calls = max(adj_stats['calls'], 1)
    norm_mean = adj_stats['norm_sum'] / calls
    norm_var = max(adj_stats['norm_sq_sum'] / calls - norm_mean * norm_mean, 0.0)
    adj_summary = {
        'calls': int(adj_stats['calls']),
        'time_ms_avg': adj_stats['time_ms_sum'] / calls,
        'norm_mean': norm_mean,
        'norm_std': float(np.sqrt(norm_var)),
        'max_abs': adj_stats['max_abs'],
        'nonfinite_count': int(adj_stats['nonfinite_count']),
        'near_zero_count': int(adj_stats['near_zero_count']),
        'delta_norm_mean': (
            adj_stats['delta_norm_sum'] / max(adj_stats['delta_norm_count'], 1)
        ),
    }

    return final_fom, total_reward, episode_steps, adj_summary, sac_steps, adjoint_steps, phase_switch_t


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
        actor_lr=args.sac_actor_lr,
        gamma=args.sac_gamma,
        tau=0.005,
        alpha_lr=args.sac_alpha_lr,
        buffer_size=args.sac_buffer_size,
        batch_size=args.sac_batch_size,
        min_buffer_size=args.sac_min_buffer_size,
        max_grad_norm=args.sac_max_grad_norm,
        target_entropy_scale=args.sac_target_entropy_scale,
        alpha_init=args.sac_alpha_init,
        reward_scale=args.sac_reward_scale,
        target_value_clip=args.sac_target_value_clip,
        actor_update_interval=args.sac_actor_update_interval,
        monitor_window=args.sac_monitor_window,
        strict_numerics=args.sac_strict_numerics,
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
    progress_path = os.path.join(log_dir, 'progress.csv')
    progress_fields = [
        'episode', 'episodes_total', 'final_fom', 'total_reward', 'steps',
        'buffer', 'episode_time_s', 'alpha', 'alpha_loss', 'critic_loss', 'actor_loss',
        'sched_delta', 'sched_phase_ratio', 'sched_target_entropy_scale',
        'q1_mean', 'q2_mean', 'q_gap', 'td_abs', 'ent', 'log_prob_mean',
        'reward_mean', 'reward_std', 'target_q_mean', 'target_q_std',
        'target_value_mean', 'target_value_std', 'done_ratio',
        'grad_actor_pre', 'grad_c1_pre', 'grad_c2_pre',
        'grad_actor', 'grad_c1', 'grad_c2',
        'clip_hit_actor', 'clip_hit_c1', 'clip_hit_c2',
        'clamp_ratio', 'apply_action_calls_total', 'apply_action_calls_ep', 'meta_coverage_ep',
        'sac_steps_ep', 'adj_steps_ep', 'phase_switch_t',
        'bad_train_tensor_total', 'bad_action_oor_total',
        'bad_nonfinite_reward_total', 'bad_nonfinite_action_total',
        'adj_calls', 'adj_time_ms_avg', 'adj_norm_mean', 'adj_norm_std',
        'adj_max_abs', 'adj_nonfinite', 'adj_near_zero', 'adj_delta_norm_mean',
    ]
    with open(log_path, 'a') as log_file, open(progress_path, 'a', newline='') as progress_file:
        progress_writer = csv.DictWriter(progress_file, fieldnames=progress_fields)
        if progress_file.tell() == 0:
            progress_writer.writeheader()
            progress_file.flush()

        log_file.write(f"\n=== SAC Pretraining Started ===\n")
        log_file.write(f"num_episodes: {args.num_episodes}\n")
        log_file.write(f"sim_type: {args.sim_type}\n")
        log_file.write(f"sac_lr: {args.sac_lr}\n")
        log_file.write(f"sac_actor_lr: {args.sac_actor_lr}\n")
        log_file.write(f"sac_delta: {args.sac_delta}\n")
        log_file.write(f"sac_patch_size: {args.sac_patch_size}\n")
        log_file.write(f"sac_batch_size: {args.sac_batch_size}\n")
        log_file.write(f"sac_min_buffer_size: {args.sac_min_buffer_size}\n")
        log_file.write(f"sac_buffer_size: {args.sac_buffer_size}\n")
        log_file.write(f"sac_gamma: {args.sac_gamma}\n")
        log_file.write(f"sac_reward_scale: {args.sac_reward_scale}\n")
        log_file.write(f"sac_target_value_clip: {args.sac_target_value_clip}\n")
        log_file.write(f"sac_actor_update_interval: {args.sac_actor_update_interval}\n")
        log_file.write(f"sac_max_grad_norm: {args.sac_max_grad_norm}\n")
        log_file.write(f"sac_target_entropy_scale: {args.sac_target_entropy_scale}\n")
        log_file.write(f"sac_target_entropy_scale_final: {args.sac_target_entropy_scale_final}\n")
        log_file.write(f"sac_alpha_init: {args.sac_alpha_init}\n")
        log_file.write(f"sac_delta_final: {args.sac_delta_final}\n")
        log_file.write(f"sac_phase_ratio_final: {args.sac_phase_ratio_final}\n")
        log_file.write(f"sac_warmup_episodes: {args.sac_warmup_episodes}\n")
        log_file.write(f"sac_schedule_mode: {args.sac_schedule_mode}\n")
        log_file.write(f"sac_monitor_window: {args.sac_monitor_window}\n")
        log_file.write(f"sac_strict_numerics: {args.sac_strict_numerics}\n")
        log_file.write(f"sac_start_ratio: {args.sac_start_ratio}\n")
        log_file.write(f"sac_phase_ratio: {args.sac_phase_ratio}\n")
        log_file.write(f"timestep_respacing: {args.timestep_respacing}\n")
        log_file.write(f"progress_csv: {progress_path}\n")
        log_file.write(f"{'='*60}\n")
        log_file.flush()

        logger.log(f"Starting SAC pretraining for {args.num_episodes} episodes...")
        logger.log(f"Log file: {log_path}")

        best_fom = -float('inf')
        best_path = None

        for episode in range(args.num_episodes):
            ep_start = time.time()
            pre_counters = dict(sac_agent.monitor_counters)

            if args.num_episodes <= 1:
                schedule_progress = 1.0
            else:
                schedule_progress = episode / float(args.num_episodes - 1)

            scheduled_delta = scheduled_value(
                args.sac_delta,
                args.sac_delta_final,
                schedule_progress,
                args.sac_schedule_mode,
            )
            scheduled_phase_ratio = scheduled_value(
                args.sac_phase_ratio,
                args.sac_phase_ratio_final,
                schedule_progress,
                args.sac_schedule_mode,
            )
            scheduled_entropy_scale = scheduled_value(
                args.sac_target_entropy_scale,
                args.sac_target_entropy_scale_final,
                schedule_progress,
                args.sac_schedule_mode,
            )

            if episode < args.sac_warmup_episodes:
                effective_phase_ratio = 0.0
            else:
                effective_phase_ratio = float(np.clip(scheduled_phase_ratio, 0.0, 1.0))

            sac_agent.set_delta(scheduled_delta)
            sac_agent.set_target_entropy_scale(scheduled_entropy_scale)

            final_fom, total_reward, ep_steps, adj_summary, sac_steps, adj_steps, phase_switch_t = run_episode(
                model, diffusion, sac_agent, sim_env,
                args, episode, device, log_file, effective_phase_ratio
            )

            ep_elapsed = time.time() - ep_start
            monitor_summary = sac_agent.get_monitor_summary()
            counters = monitor_summary.get('counters', {})
            clamp_ratio = counters.get('apply_action_clamp_ratio', 0.0)
            nonfinite_train = counters.get('nonfinite_train_tensor', 0)
            action_oor = counters.get('out_of_range_action', 0)
            nonfinite_reward = counters.get('nonfinite_reward', 0)
            nonfinite_action = counters.get('nonfinite_action', 0)
            apply_calls_total = int(counters.get('apply_action_calls', 0))
            apply_calls_ep = apply_calls_total - int(pre_counters.get('apply_action_calls', 0))
            meta_coverage_ep = apply_calls_ep / max(diffusion.num_timesteps, 1)

            critic_loss = monitor_summary.get('critic_loss', 0.0)
            actor_loss = monitor_summary.get('actor_loss', 0.0)
            alpha_loss = monitor_summary.get('alpha_loss', 0.0)
            q1_mean = monitor_summary.get('q1_mean', 0.0)
            q2_mean = monitor_summary.get('q2_mean', 0.0)
            q_gap = monitor_summary.get('q_gap_abs_mean', 0.0)
            td_abs = monitor_summary.get('td_error_abs_mean', 0.0)
            entropy = monitor_summary.get('entropy_estimate', 0.0)
            log_prob_mean = monitor_summary.get('log_prob_mean', 0.0)
            reward_mean = monitor_summary.get('reward_mean', 0.0)
            reward_std = monitor_summary.get('reward_std', 0.0)
            target_q_mean = monitor_summary.get('target_q_mean', 0.0)
            target_q_std = monitor_summary.get('target_q_std', 0.0)
            target_value_mean = monitor_summary.get('target_value_mean', 0.0)
            target_value_std = monitor_summary.get('target_value_std', 0.0)
            done_ratio = monitor_summary.get('done_ratio', 0.0)
            grad_actor = monitor_summary.get('actor_grad_norm', 0.0)
            grad_c1 = monitor_summary.get('critic1_grad_norm', 0.0)
            grad_c2 = monitor_summary.get('critic2_grad_norm', 0.0)
            grad_actor_pre = monitor_summary.get('actor_grad_norm_pre', grad_actor)
            grad_c1_pre = monitor_summary.get('critic1_grad_norm_pre', grad_c1)
            grad_c2_pre = monitor_summary.get('critic2_grad_norm_pre', grad_c2)
            clip_hit_actor = monitor_summary.get('actor_clip_hit', 0.0)
            clip_hit_c1 = monitor_summary.get('critic1_clip_hit', 0.0)
            clip_hit_c2 = monitor_summary.get('critic2_clip_hit', 0.0)

            # Episode summary
            ep_log = (f"EP {episode:3d}/{args.num_episodes} | "
                      f"final_fom={final_fom:.6f} | "
                      f"total_reward={total_reward:.6f} | "
                      f"steps={ep_steps} | "
                      f"buffer={len(sac_agent.replay_buffer)} | "
                      f"alpha={sac_agent.alpha.item():.4f} | "
                      f"sched(d/p/e)={scheduled_delta:.4f}/{effective_phase_ratio:.3f}/{scheduled_entropy_scale:.3f} | "
                      f"actor_loss={actor_loss:.4f} | alpha_loss={alpha_loss:.4f} | q_gap={q_gap:.4f} | td_abs={td_abs:.4f} | ent={entropy:.4f} | "
                      f"clamp={clamp_ratio:.3f} | bad(train/act)={nonfinite_train}/{action_oor} | "
                      f"phase(sac/adj/sw)={sac_steps}/{adj_steps}/{phase_switch_t:.1f} | "
                      f"meta={apply_calls_ep}/{diffusion.num_timesteps} ({meta_coverage_ep:.2f}) | "
                      f"adj(ms/norm/max)={adj_summary['time_ms_avg']:.1f}/{adj_summary['norm_mean']:.3e}/{adj_summary['max_abs']:.3e} | "
                      f"adj_bad={adj_summary['nonfinite_count']}/{adj_summary['near_zero_count']} | "
                      f"time={ep_elapsed:.1f}s")
            print(f"\n{'='*70}")
            print(ep_log)
            print(f"{'='*70}\n")
            log_file.write(f"\n{ep_log}\n")
            log_file.flush()

            progress_writer.writerow({
                'episode': episode,
                'episodes_total': args.num_episodes,
                'final_fom': f"{final_fom:.6f}",
                'total_reward': f"{total_reward:.6f}",
                'steps': ep_steps,
                'buffer': len(sac_agent.replay_buffer),
                'episode_time_s': f"{ep_elapsed:.2f}",
                'alpha': f"{sac_agent.alpha.item():.6f}",
                'alpha_loss': f"{alpha_loss:.6f}",
                'critic_loss': f"{critic_loss:.6f}",
                'actor_loss': f"{actor_loss:.6f}",
                'sched_delta': f"{scheduled_delta:.6f}",
                'sched_phase_ratio': f"{effective_phase_ratio:.6f}",
                'sched_target_entropy_scale': f"{scheduled_entropy_scale:.6f}",
                'q1_mean': f"{q1_mean:.6f}",
                'q2_mean': f"{q2_mean:.6f}",
                'q_gap': f"{q_gap:.6f}",
                'td_abs': f"{td_abs:.6f}",
                'ent': f"{entropy:.6f}",
                'log_prob_mean': f"{log_prob_mean:.6f}",
                'reward_mean': f"{reward_mean:.6f}",
                'reward_std': f"{reward_std:.6f}",
                'target_q_mean': f"{target_q_mean:.6f}",
                'target_q_std': f"{target_q_std:.6f}",
                'target_value_mean': f"{target_value_mean:.6f}",
                'target_value_std': f"{target_value_std:.6f}",
                'done_ratio': f"{done_ratio:.6f}",
                'grad_actor_pre': f"{grad_actor_pre:.6f}",
                'grad_c1_pre': f"{grad_c1_pre:.6f}",
                'grad_c2_pre': f"{grad_c2_pre:.6f}",
                'grad_actor': f"{grad_actor:.6f}",
                'grad_c1': f"{grad_c1:.6f}",
                'grad_c2': f"{grad_c2:.6f}",
                'clip_hit_actor': f"{clip_hit_actor:.6f}",
                'clip_hit_c1': f"{clip_hit_c1:.6f}",
                'clip_hit_c2': f"{clip_hit_c2:.6f}",
                'clamp_ratio': f"{clamp_ratio:.6f}",
                'apply_action_calls_total': apply_calls_total,
                'apply_action_calls_ep': apply_calls_ep,
                'meta_coverage_ep': f"{meta_coverage_ep:.6f}",
                'sac_steps_ep': sac_steps,
                'adj_steps_ep': adj_steps,
                'phase_switch_t': f"{phase_switch_t:.6f}",
                'bad_train_tensor_total': nonfinite_train,
                'bad_action_oor_total': action_oor,
                'bad_nonfinite_reward_total': nonfinite_reward,
                'bad_nonfinite_action_total': nonfinite_action,
                'adj_calls': adj_summary['calls'],
                'adj_time_ms_avg': f"{adj_summary['time_ms_avg']:.6f}",
                'adj_norm_mean': f"{adj_summary['norm_mean']:.6f}",
                'adj_norm_std': f"{adj_summary['norm_std']:.6f}",
                'adj_max_abs': f"{adj_summary['max_abs']:.6f}",
                'adj_nonfinite': adj_summary['nonfinite_count'],
                'adj_near_zero': adj_summary['near_zero_count'],
                'adj_delta_norm_mean': f"{adj_summary['delta_norm_mean']:.6f}",
            })
            progress_file.flush()

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
        sac_lr=1e-4,
        sac_actor_lr=2e-5,
        sac_alpha_lr=5e-4,
        sac_delta=0.04,
        sac_patch_size=8,
        sac_batch_size=64,
        sac_min_buffer_size=512,
        sac_buffer_size=50000,
        sac_gamma=0.99,
        sac_reward_scale=1.0,
        sac_target_value_clip=10.0,
        sac_actor_update_interval=3,
        sac_max_grad_norm=5.0,
        sac_target_entropy_scale=0.05,
        sac_target_entropy_scale_final=0.10,
        sac_alpha_init=0.05,
        sac_start_ratio=0.5,
        sac_phase_ratio=0.35,
        sac_phase_ratio_final=0.65,
        sac_delta_final=0.08,
        sac_warmup_episodes=4,
        sac_schedule_mode="linear",
        sac_monitor_window=200,
        sac_strict_numerics=False,
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
