"""
SAC (Soft Actor-Critic) Agent for guiding diffusion model sampling.

Replaces DQN with a continuous action space RL approach:
- State: (pred_xstart, adjoint_gradient, timestep) — 3-channel 64x64 image
- Action: continuous patch-level perturbation (num_patches dimensions)
- Reward: change in figure of merit (fom)

Key improvements over DQN:
1. Continuous action space — all patches modified simultaneously
2. Adjoint gradient as auxiliary state input — leverages simulator info
3. Entropy-regularized — encourages exploration via automatic temperature
"""

import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class CNNEncoder(nn.Module):
    """
    Shared CNN encoder for extracting features from state images.

    Input: (batch, in_channels, 64, 64)
        - channel 0: pred_xstart (current design)
        - channel 1: adjoint gradient (sensitivity map)
        - channel 2: timestep map (scalar t/T broadcast to 64x64)
    Output: (batch, feature_dim) feature vector
    """

    def __init__(self, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # 16 -> 8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),                                  # -> 4x4
        )
        self.feature_dim = 128 * 4 * 4  # 2048

    def forward(self, x):
        features = self.features(x)
        return features.view(features.size(0), -1)


class SACActorNetwork(nn.Module):
    """
    Stochastic Actor: maps state to a Gaussian distribution over actions.

    Input: (batch, 3, 64, 64) state image
    Output: mean, log_std of shape (batch, num_patches)
    Actions are squashed through tanh to [-1, 1].
    """

    def __init__(self, image_size=64, patch_size=8, in_channels=3):
        super().__init__()
        self.num_patches_per_dim = image_size // patch_size
        self.num_patches = self.num_patches_per_dim ** 2  # e.g. 64
        self.action_dim = self.num_patches

        self.encoder = CNNEncoder(in_channels)

        self.mlp = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(256, self.action_dim)
        self.log_std_head = nn.Linear(256, self.action_dim)

    def forward(self, state):
        """
        Args:
            state: (batch, 3, 64, 64) tensor
        Returns:
            mean: (batch, action_dim)
            log_std: (batch, action_dim)
        """
        features = self.encoder(state)
        h = self.mlp(features)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        """
        Sample action with reparameterization trick + tanh squashing.

        Returns:
            action: (batch, action_dim) in [-1, 1]
            log_prob: (batch,) log probability
            mean: (batch, action_dim) deterministic action
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        # Reparameterization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        # Log probability with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        return action, log_prob, torch.tanh(mean)


class SACCriticNetwork(nn.Module):
    """
    Q-Network: maps (state, action) to a scalar Q-value.

    Input: state (batch, 3, 64, 64), action (batch, action_dim)
    Output: Q-value (batch, 1)
    """

    def __init__(self, image_size=64, patch_size=8, in_channels=3):
        super().__init__()
        self.num_patches_per_dim = image_size // patch_size
        self.num_patches = self.num_patches_per_dim ** 2
        self.action_dim = self.num_patches

        self.encoder = CNNEncoder(in_channels)

        # Project action to a hidden dimension
        self.action_proj = nn.Sequential(
            nn.Linear(self.action_dim, 256),
            nn.ReLU(),
        )

        # Combine state features + action features
        self.q_head = nn.Sequential(
            nn.Linear(self.encoder.feature_dim + 256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        """
        Args:
            state: (batch, 3, 64, 64)
            action: (batch, action_dim)
        Returns:
            q_value: (batch, 1)
        """
        state_features = self.encoder(state)
        action_features = self.action_proj(action)
        combined = torch.cat([state_features, action_features], dim=-1)
        return self.q_head(combined)


class ReplayBuffer:
    """Experience replay buffer for SAC training with adjoint gradient support."""

    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, adjoint_grad, timestep, action, reward,
             next_state, next_adjoint_grad, next_timestep, done=False):
        """Store a transition."""
        self.buffer.append((
            state.detach().cpu(),
            adjoint_grad.detach().cpu(),
            timestep,
            action.detach().cpu() if torch.is_tensor(action) else torch.tensor(action, dtype=torch.float32),
            reward,
            next_state.detach().cpu(),
            next_adjoint_grad.detach().cpu(),
            next_timestep,
            done,
        ))

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        (states, adj_grads, timesteps, actions, rewards,
         next_states, next_adj_grads, next_timesteps, dones) = zip(*batch)

        states = torch.cat(states, dim=0)
        adj_grads = torch.cat(adj_grads, dim=0)
        timesteps = torch.tensor(timesteps, dtype=torch.float32)
        actions = torch.stack(actions, dim=0)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.cat(next_states, dim=0)
        next_adj_grads = torch.cat(next_adj_grads, dim=0)
        next_timesteps = torch.tensor(next_timesteps, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        return (states, adj_grads, timesteps, actions, rewards,
                next_states, next_adj_grads, next_timesteps, dones)

    def __len__(self):
        return len(self.buffer)


class SACAgent:
    """
    SAC Agent that guides diffusion sampling by selecting
    continuous patch-level modifications on the predicted x0.

    Key advantages over DQN:
    - Continuous actions: all patches modified simultaneously
    - Adjoint gradient as input: leverages simulator sensitivity info
    - Entropy bonus: automatic exploration via temperature tuning
    """

    def __init__(
        self,
        image_size=64,
        patch_size=8,
        delta=0.1,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha_lr=3e-4,
        buffer_size=50000,
        batch_size=256,
        target_entropy=None,
        device=None,
    ):
        """
        Args:
            image_size: size of the square design image (default 64)
            patch_size: size of square patches for action space (default 8)
            delta: perturbation magnitude scaling factor
            lr: learning rate for actor and critic
            gamma: discount factor
            tau: soft update coefficient for target networks
            alpha_lr: learning rate for temperature parameter
            buffer_size: replay buffer capacity
            batch_size: batch size for training
            target_entropy: target entropy for automatic temperature tuning
            device: torch device
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.delta = delta
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_patches_per_dim = image_size // patch_size
        self.num_patches = self.num_patches_per_dim ** 2
        self.action_dim = self.num_patches

        # Actor
        self.actor = SACActorNetwork(image_size, patch_size, in_channels=3).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Twin Critics
        self.critic1 = SACCriticNetwork(image_size, patch_size, in_channels=3).to(self.device)
        self.critic2 = SACCriticNetwork(image_size, patch_size, in_channels=3).to(self.device)
        self.critic1_target = SACCriticNetwork(image_size, patch_size, in_channels=3).to(self.device)
        self.critic2_target = SACCriticNetwork(image_size, patch_size, in_channels=3).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic1_target.eval()
        self.critic2_target.eval()
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr
        )

        # Automatic temperature tuning
        if target_entropy is None:
            self.target_entropy = -self.action_dim  # heuristic: -dim(A)
        else:
            self.target_entropy = target_entropy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        self.train_step_count = 0
        self.total_critic_loss = 0.0
        self.total_actor_loss = 0.0
        self.loss_count = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _build_state(self, pred_xstart, adjoint_grad, timestep):
        """
        Build 3-channel state tensor from components.

        Args:
            pred_xstart: (1, 1, 64, 64) tensor in [-1, 1]
            adjoint_grad: (1, 1, 64, 64) tensor (adjoint gradient / sensitivity)
            timestep: float in [0, 1], normalized timestep

        Returns:
            state: (1, 3, 64, 64) tensor
        """
        # Normalize adjoint gradient to [-1, 1] range
        adj_max = adjoint_grad.abs().max()
        if adj_max > 0:
            adjoint_grad_norm = adjoint_grad / adj_max
        else:
            adjoint_grad_norm = adjoint_grad

        # Create timestep channel: constant map
        t_channel = torch.full_like(pred_xstart, timestep)

        # Concatenate: (1, 3, 64, 64)
        state = torch.cat([pred_xstart, adjoint_grad_norm, t_channel], dim=1)
        return state

    def select_action(self, pred_xstart, adjoint_grad, timestep, deterministic=False):
        """
        Select an action using the actor policy.

        Args:
            pred_xstart: (1, 1, 64, 64) tensor
            adjoint_grad: (1, 1, 64, 64) tensor
            timestep: float in [0, 1]
            deterministic: if True, use mean action (no exploration)

        Returns:
            action: (num_patches,) numpy array in [-1, 1]
        """
        state = self._build_state(pred_xstart, adjoint_grad, timestep)
        state = state.to(self.device)

        with torch.no_grad():
            if deterministic:
                _, _, action = self.actor.sample(state)
            else:
                action, _, _ = self.actor.sample(state)

        return action.squeeze(0).cpu()

    def apply_action(self, pred_xstart, action):
        """
        Apply the continuous action to pred_xstart.

        Each action value controls the perturbation for a patch:
            patch[i] += action[i] * delta

        All patches are modified simultaneously (unlike DQN which modifies only one).

        Args:
            pred_xstart: (1, 1, 64, 64) tensor
            action: (num_patches,) tensor in [-1, 1]

        Returns:
            modified pred_xstart (clamped to [-1, 1])
        """
        modified = pred_xstart.clone()
        action = action.to(pred_xstart.device)

        for i in range(self.num_patches):
            row = i // self.num_patches_per_dim
            col = i % self.num_patches_per_dim
            r_start = row * self.patch_size
            r_end = r_start + self.patch_size
            c_start = col * self.patch_size
            c_end = c_start + self.patch_size
            modified[:, :, r_start:r_end, c_start:c_end] += action[i].item() * self.delta

        modified = modified.clamp(-1, 1)
        return modified

    def store_transition(self, pred_xstart, adjoint_grad, timestep, action,
                         reward, next_pred_xstart, next_adjoint_grad, next_timestep,
                         done=False):
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(
            pred_xstart, adjoint_grad, timestep, action, reward,
            next_pred_xstart, next_adjoint_grad, next_timestep, done
        )

    def train_step(self):
        """
        Perform one training step of SAC.

        Updates: critics → actor → temperature (alpha).
        Uses torch.enable_grad() because p_sample runs under no_grad().

        Returns:
            dict with 'critic_loss', 'actor_loss', 'alpha', 'alpha_loss'
            or None if buffer too small
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        (states, adj_grads, timesteps, actions, rewards,
         next_states, next_adj_grads, next_timesteps, dones) = self.replay_buffer.sample(
            self.batch_size
        )

        # Move to device
        states = states.to(self.device)
        adj_grads = adj_grads.to(self.device)
        timesteps = timesteps.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        next_adj_grads = next_adj_grads.to(self.device)
        next_timesteps = next_timesteps.to(self.device)
        dones = dones.to(self.device)

        # Build 3-channel state tensors
        # states are (B, 1, 64, 64), adj_grads are (B, 1, 64, 64)
        # Need to normalize adj_grads per-sample
        adj_max = adj_grads.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
        adj_grads_norm = adj_grads / adj_max
        t_channels = timesteps.view(-1, 1, 1, 1).expand_as(states)
        state_3ch = torch.cat([states, adj_grads_norm, t_channels], dim=1)

        next_adj_max = next_adj_grads.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
        next_adj_grads_norm = next_adj_grads / next_adj_max
        next_t_channels = next_timesteps.view(-1, 1, 1, 1).expand_as(next_states)
        next_state_3ch = torch.cat([next_states, next_adj_grads_norm, next_t_channels], dim=1)

        with torch.enable_grad():
            # ---- Update Critics ----
            with torch.no_grad():
                next_actions, next_log_probs, _ = self.actor.sample(next_state_3ch)
                target_q1 = self.critic1_target(next_state_3ch, next_actions)
                target_q2 = self.critic2_target(next_state_3ch, next_actions)
                target_q = torch.min(target_q1, target_q2).squeeze(-1)
                target_q = target_q - self.alpha.detach() * next_log_probs
                target_value = rewards + self.gamma * (1 - dones) * target_q

            current_q1 = self.critic1(state_3ch, actions).squeeze(-1)
            current_q2 = self.critic2(state_3ch, actions).squeeze(-1)

            critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.critic1.parameters()) + list(self.critic2.parameters()),
                max_norm=1.0
            )
            self.critic_optimizer.step()

            # ---- Update Actor ----
            new_actions, log_probs, _ = self.actor.sample(state_3ch.detach())
            q1_new = self.critic1(state_3ch.detach(), new_actions)
            q2_new = self.critic2(state_3ch.detach(), new_actions)
            q_new = torch.min(q1_new, q2_new).squeeze(-1)

            actor_loss = (self.alpha.detach() * log_probs - q_new).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            # ---- Update Temperature ----
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # Soft update target networks
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        self.train_step_count += 1
        critic_loss_val = critic_loss.item()
        actor_loss_val = actor_loss.item()
        self.total_critic_loss += critic_loss_val
        self.total_actor_loss += actor_loss_val
        self.loss_count += 1

        return {
            'critic_loss': critic_loss_val,
            'actor_loss': actor_loss_val,
            'alpha': self.alpha.item(),
            'alpha_loss': alpha_loss.item(),
        }

    def _soft_update(self, source, target):
        """Polyak averaging for target network update."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def get_avg_loss(self):
        """Get average training losses since last call."""
        if self.loss_count == 0:
            return 0.0, 0.0
        avg_critic = self.total_critic_loss / self.loss_count
        avg_actor = self.total_actor_loss / self.loss_count
        self.total_critic_loss = 0.0
        self.total_actor_loss = 0.0
        self.loss_count = 0
        return avg_critic, avg_actor

    def save(self, path):
        """Save agent state."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'train_step_count': self.train_step_count,
        }, path)

    def load(self, path):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.log_alpha = checkpoint['log_alpha'].to(self.device).requires_grad_(True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_optimizer.defaults['lr'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        self.train_step_count = checkpoint.get('train_step_count', 0)
