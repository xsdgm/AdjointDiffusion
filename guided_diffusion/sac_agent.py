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


class SpatialAttention(nn.Module):
    """
    Self-attention mechanism across spatial patches to capture long-range dependencies.
    """
    def __init__(self, d_model=128, nhead=4, num_layers=1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=0.0, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # Flatten spatial dims to sequence: (B, H*W, C)
        x_seq = x.view(B, C, H * W).permute(0, 2, 1)
        # Apply transformer
        out_seq = self.transformer(x_seq)
        # Reshape back to (B, C, H, W)
        out = out_seq.permute(0, 2, 1).view(B, C, H, W)
        return out


class CNNEncoder(nn.Module):
    """
    Shared CNN encoder with Self-Attention for extracting spatial features.

    Input: (batch, in_channels, 64, 64)
    Output: (batch, 128, 8, 8) feature map (assuming patch_size=8)
    """

    def __init__(self, in_channels=3, use_attention=True):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # 16 -> 8
            nn.ReLU(inplace=True),
        )
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = SpatialAttention(d_model=128, nhead=4, num_layers=1)
        
        self.feature_channels = 128

    def forward(self, x):
        feat = self.features(x)
        if self.use_attention:
            feat = self.attention(feat)
        return feat


class SACActorNetwork(nn.Module):
    """
    Stochastic Actor: maps state to a Gaussian distribution over actions.
    Uses fully convolutional action heads to keep strict spatial correlation.

    Input: (batch, 3, 64, 64) state image
    Output: mean, log_std of shape (batch, num_patches)
    Actions are squashed through tanh to [-1, 1].
    """

    def __init__(self, image_size=64, patch_size=8, in_channels=3):
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")
        if image_size % patch_size != 0:
            raise ValueError(f"image_size ({image_size}) must be divisible by patch_size ({patch_size})")
        self.num_patches_per_dim = image_size // patch_size
        self.num_patches = self.num_patches_per_dim ** 2
        self.action_dim = self.num_patches + 1  # spatial patches + eta gate

        self.encoder = CNNEncoder(in_channels, use_attention=True)

        # 1x1 Convolutions for pixel-to-patch spatial alignment
        self.mean_head = nn.Conv2d(self.encoder.feature_channels, 1, kernel_size=1)
        self.log_std_head = nn.Conv2d(self.encoder.feature_channels, 1, kernel_size=1)
        
        # Scalar head for adaptive eta gating (blending with adjoint gradient)
        self.eta_mean_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.encoder.feature_channels, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
        self.eta_log_std_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.encoder.feature_channels, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, state):
        """
        Args:
            state: (batch, 3, 64, 64) tensor
        Returns:
            mean: (batch, action_dim) [patch_actions, eta_gate]
            log_std: (batch, action_dim)
        """
        features = self.encoder(state) # (B, 128, 8, 8)
        
        # Spatial action map is resized to patch grid so action_dim follows patch_size.
        patch_mean_map = self.mean_head(features)
        patch_log_std_map = self.log_std_head(features)
        if patch_mean_map.shape[-2:] != (self.num_patches_per_dim, self.num_patches_per_dim):
            patch_mean_map = F.interpolate(
                patch_mean_map,
                size=(self.num_patches_per_dim, self.num_patches_per_dim),
                mode="bilinear",
                align_corners=False,
            )
            patch_log_std_map = F.interpolate(
                patch_log_std_map,
                size=(self.num_patches_per_dim, self.num_patches_per_dim),
                mode="bilinear",
                align_corners=False,
            )

        patch_mean = patch_mean_map.view(features.size(0), -1)
        patch_log_std = patch_log_std_map.view(features.size(0), -1)
        
        # Scalar action: Output (B, 1)
        eta_mean = self.eta_mean_head(features)
        eta_log_std = self.eta_log_std_head(features)
        
        # Concatenate actions: (B, 65)
        mean = torch.cat([patch_mean, eta_mean], dim=1)
        log_std = torch.cat([patch_log_std, eta_log_std], dim=1)
        
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

        # Log probability with tanh correction using numerically stable softplus formulation
        log_prob = normal.log_prob(x_t)
        log_prob -= (2 * (np.log(2) - x_t - F.softplus(-2 * x_t)))
        log_prob = log_prob.sum(dim=-1)

        return action, log_prob, torch.tanh(mean)


class SACCriticNetwork(nn.Module):
    """
    Q-Network: maps (state, action) to a scalar Q-value.
    Leverages spatial CNN by mapping action back to 2D grid.

    Input: state (batch, 3, 64, 64), action (batch, action_dim)
    Output: Q-value (batch, 1)
    """

    def __init__(self, image_size=64, patch_size=8, in_channels=3):
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")
        if image_size % patch_size != 0:
            raise ValueError(f"image_size ({image_size}) must be divisible by patch_size ({patch_size})")
        self.num_patches_per_dim = image_size // patch_size
        self.num_patches = self.num_patches_per_dim ** 2
        self.action_dim = self.num_patches + 1  # 64 spatial + 1 eta scalar

        self.encoder = CNNEncoder(in_channels, use_attention=True)

        # Add action channels (1) to state channels (128) -> 129
        # Downsample 8x8 -> 1 scalar value
        self.q_conv = nn.Sequential(
            nn.Conv2d(self.encoder.feature_channels + 1, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1), # 8 -> 4
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),  # 4 -> 2
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        # Final linear layers process flattened conv output + scalar eta
        self.q_linear = nn.Sequential(
            nn.Linear(32 * 2 * 2 + 1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        """
        Args:
            state: (batch, 3, 64, 64)
            action: (batch, action_dim) -> (batch, 65)
        Returns:
            q_value: (batch, 1)
        """
        state_features = self.encoder(state) # (B, 128, 8, 8)
        
        # Split action into patch actions (64) and eta (1)
        patch_actions = action[:, :self.num_patches_per_dim**2]
        eta_action = action[:, self.num_patches_per_dim**2:]
        
        # Reshape patch action to patch grid and resize to critic feature map resolution.
        action_map = patch_actions.view(-1, 1, self.num_patches_per_dim, self.num_patches_per_dim)
        if action_map.shape[-2:] != state_features.shape[-2:]:
            action_map = F.interpolate(
                action_map,
                size=state_features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        
        # Concatenate spatial components -> (B, 129, 8, 8)
        combined_spatial = torch.cat([state_features, action_map], dim=1)
        
        conv_out = self.q_conv(combined_spatial) # (B, 128)
        
        # Concatenate flat conv out with eta scalar -> (B, 129)
        combined_flat = torch.cat([conv_out, eta_action], dim=1)
        
        return self.q_linear(combined_flat)


class RewardNormalizer:
    """Running reward normalization using Welford's online algorithm.

    Normalizes rewards to zero-mean, unit-variance using an exponentially
    weighted running estimate. This replaces hardcoded reward scaling and
    adapts automatically to the magnitude of FoM differences.
    """

    def __init__(self, clip_range=10.0):
        self.mean = 0.0
        self.m2 = 0.0
        self.var = 1.0
        self.count = 0
        self.clip_range = clip_range

    def update(self, reward):
        """Update running statistics with one raw reward sample."""
        self.count += 1
        if self.count == 1:
            self.mean = reward
            self.m2 = 0.0
            self.var = 1.0
        else:
            delta = reward - self.mean
            self.mean += delta / self.count
            delta2 = reward - self.mean
            self.m2 += delta * delta2
            self.var = max(self.m2 / self.count, 1e-8)

    def _normalize_value(self, reward):
        if self.count <= 1:
            return 0.0

        std = max(self.var ** 0.5, 1e-8)
        normalized = (reward - self.mean) / std
        return max(-self.clip_range, min(self.clip_range, normalized))

    def normalize(self, reward, update=True):
        """Return a normalized reward, optionally updating running statistics first."""
        if update:
            self.update(reward)

        return self._normalize_value(reward)

    def normalize_tensor(self, rewards):
        """Normalize a tensor of rewards using the current frozen statistics."""
        if self.count <= 1:
            return torch.zeros_like(rewards)

        std = max(self.var ** 0.5, 1e-8)
        normalized = (rewards - self.mean) / std
        return normalized.clamp(-self.clip_range, self.clip_range)

    def state_dict(self):
        return {'mean': self.mean, 'm2': self.m2, 'var': self.var, 'count': self.count,
                'clip_range': self.clip_range}

    def load_state_dict(self, d):
        self.mean = d['mean']
        self.count = d['count']
        self.m2 = d.get('m2')
        if self.m2 is None:
            loaded_var = d.get('var', 1.0)
            self.m2 = max(float(loaded_var), 0.0) * max(self.count, 1)
        self.var = max(d.get('var', self.m2 / max(self.count, 1)), 1e-8)
        self.clip_range = d.get('clip_range', 10.0)


class ReplayBuffer:
    """Experience replay buffer for SAC training with adjoint gradient support."""

    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    @staticmethod
    def _normalize_scalar(value):
        if torch.is_tensor(value):
            return value.detach().cpu().item()
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _normalize_tensor(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        return torch.as_tensor(value, dtype=torch.float32)

    @classmethod
    def _normalize_transition(cls, transition):
        state, adjoint_grad, timestep, action, reward, next_state, next_adjoint_grad, next_timestep, done = transition
        return (
            cls._normalize_tensor(state),
            cls._normalize_tensor(adjoint_grad),
            cls._normalize_scalar(timestep),
            cls._normalize_tensor(action),
            cls._normalize_scalar(reward),
            cls._normalize_tensor(next_state),
            cls._normalize_tensor(next_adjoint_grad),
            cls._normalize_scalar(next_timestep),
            bool(cls._normalize_scalar(done)),
        )

    def push(self, state, adjoint_grad, timestep, action, reward,
             next_state, next_adjoint_grad, next_timestep, done=False):
        """Store a transition."""
        self.buffer.append(self._normalize_transition((
            state,
            adjoint_grad,
            timestep,
            action,
            reward,
            next_state,
            next_adjoint_grad,
            next_timestep,
            done,
        )))

    def sample(self, batch_size):
        """Sample a batch of transitions. Caller must ensure len(buffer) >= batch_size."""
        batch = random.sample(self.buffer, batch_size)
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

    def state_dict(self):
        return {
            'capacity': self.buffer.maxlen,
            'buffer': list(self.buffer),
        }

    def load_state_dict(self, d):
        capacity = d.get('capacity', len(d.get('buffer', [])) or 1)
        buffer = d.get('buffer', [])
        normalized_buffer = [self._normalize_transition(transition) for transition in buffer]
        self.buffer = deque(normalized_buffer, maxlen=capacity)


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
        batch_size=64,
        target_entropy=None,
        reward_scale=1.0,
        min_buffer_size=None,
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
            reward_scale: fixed multiplier for raw rewards (default 1.0)
            min_buffer_size: minimum transitions before training starts
                             (default: batch_size * 4)
            device: torch device
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.delta = delta
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.min_buffer_size = min_buffer_size if min_buffer_size is not None else batch_size * 4
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")
        if image_size % patch_size != 0:
            raise ValueError(f"image_size ({image_size}) must be divisible by patch_size ({patch_size})")

        self.num_patches_per_dim = image_size // patch_size
        self.num_patches = self.num_patches_per_dim ** 2
        self.action_dim = self.num_patches + 1

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
        # Independent optimizers so Adam's moment estimates do not cross-contaminate
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        # Automatic temperature tuning
        if target_entropy is None:
            self.target_entropy = -float(self.action_dim)  # -dim(A) heuristic
        else:
            self.target_entropy = target_entropy
        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=self.device)  # init alpha=1.0
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        self.reward_normalizer = RewardNormalizer()

        self.train_step_count = 0
        self.total_critic_loss = 0.0
        self.total_actor_loss = 0.0
        self.loss_count = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def normalize_reward(self, raw_reward):
        """Normalize a raw reward using running statistics.

        Call this instead of hardcoded scaling (e.g. ``reward * 100``).
        """
        return self.reward_normalizer.normalize(raw_reward)

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
        # Normalize adjoint gradient to [-1, 1] range (consistent with train_step)
        adj_max = adjoint_grad.abs().amax().clamp(min=1e-8)
        adjoint_grad_norm = adjoint_grad / adj_max

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
            action: (action_dim,) numpy array in [-1, 1]
        """
        state = self._build_state(pred_xstart, adjoint_grad, timestep)
        state = state.to(self.device)

        with torch.no_grad():
            if deterministic:
                _, _, action = self.actor.sample(state)
            else:
                action, _, _ = self.actor.sample(state)

        action = action.cpu()
        if action.shape[0] == 1:
            return action.squeeze(0)
        return action

    def apply_action(self, pred_xstart, action, adjoint_grad=None):
        """
        Apply the continuous action to pred_xstart using Adaptive Blending (Meta-Control).

        - The first 64 values control the RL perturbation for a spatial patch.
        - The 65th value is squashed to [0, 1] as an `eta_weight` gating scalar.
        
        Final update = (1 - eta_weight) * (RL Patch Action) + (eta_weight) * (Physics Adjoint Action)

        Args:
            pred_xstart: (1, 1, image_size, image_size) tensor
            action: (1 + num_patches,) tensor in [-1, 1]
            adjoint_grad: (1, 1, image_size, image_size) tensor from physics simulator

        Returns:
            modified pred_xstart (clamped to [-1, 1])
        """
        action = action.to(pred_xstart.device)

        squeeze_action = False
        if action.dim() == 1:
            action = action.unsqueeze(0)
            squeeze_action = True

        batch_size = pred_xstart.shape[0]
        if action.shape[0] != batch_size:
            raise ValueError(
                f"Action batch size ({action.shape[0]}) must match pred_xstart batch size ({batch_size})"
            )

        patch_actions = action[:, :self.num_patches]
        eta_action = action[:, self.num_patches:self.num_patches + 1]

        # Squash eta to [0, 1]
        eta_weight = ((eta_action + 1.0) / 2.0).view(batch_size, 1, 1, 1)

        # 1. RL Spatial Action Map
        # (B, num_patches) -> (B, 1, grid, grid) -> (B, 1, image_size, image_size)
        rl_action_map = patch_actions.view(batch_size, 1, self.num_patches_per_dim, self.num_patches_per_dim)
        rl_action_map = F.interpolate(
            rl_action_map,
            size=pred_xstart.shape[-2:],
            mode="nearest",
        )

        # 2. Physics Gradient Map (with heuristic scale factor)
        physics_action_map = torch.zeros_like(pred_xstart)
        if adjoint_grad is not None:
            physics_action_map = adjoint_grad.to(pred_xstart.device)
            # Normalize gradients per instance for stable combination
            max_grad = physics_action_map.abs().amax(dim=(1,2,3), keepdim=True).clamp(min=1e-8)
            physics_action_map = physics_action_map / max_grad

        # 3. Adaptive Blending
        blended_action = (1.0 - eta_weight) * rl_action_map * self.delta + (eta_weight) * physics_action_map * self.delta

        modified = pred_xstart + blended_action
        modified = modified.clamp(-1, 1)
        if squeeze_action and batch_size == 1:
            return modified
        return modified

    def store_transition(self, pred_xstart, adjoint_grad, timestep, action,
                         reward, next_pred_xstart, next_adjoint_grad, next_timestep,
                         done=False):
        """Store a transition in the replay buffer.

        Raw rewards are stored unchanged so all replayed samples keep the same
        semantics. Running statistics are still updated online and are applied
        later during training when a batch is sampled.
        """
        self.reward_normalizer.update(reward)
        self.replay_buffer.push(
            pred_xstart, adjoint_grad, timestep, action, reward,
            next_pred_xstart, next_adjoint_grad, next_timestep, done
        )

    def _set_critic_grad_enabled(self, enabled):
        """Enable or disable gradients for the online critics during actor updates."""
        for critic in (self.critic1, self.critic2):
            for param in critic.parameters():
                param.requires_grad_(enabled)

    def train_step(self):
        """
        Perform one training step of SAC.

        Updates: critics → actor → temperature (alpha).
        Uses torch.enable_grad() because p_sample runs under no_grad().

        Returns:
            dict with 'critic_loss', 'actor_loss', 'alpha', 'alpha_loss'
            or None if buffer too small
        """
        if len(self.replay_buffer) < self.min_buffer_size:
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

        # Rewards are stored raw in replay; normalize them against the current
        # frozen running statistics so every sampled batch uses one consistent scale.
        rewards = self.reward_normalizer.normalize_tensor(rewards) * self.reward_scale

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

            critic1_loss = F.mse_loss(current_q1, target_value)
            critic2_loss = F.mse_loss(current_q2, target_value)

            # Backward separately to avoid double-counting gradients
            # if a shared encoder is ever introduced
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
            self.critic2_optimizer.step()

            critic_loss = critic1_loss + critic2_loss  # for logging only

            # ---- Update Actor ----
            # Detach state_3ch so critic gradients from backward() above
            # do not leak into the actor's computation graph.
            state_3ch_detached = state_3ch.detach()
            self._set_critic_grad_enabled(False)
            try:
                new_actions, log_probs, _ = self.actor.sample(state_3ch_detached)
                q1_new = self.critic1(state_3ch_detached, new_actions)
                q2_new = self.critic2(state_3ch_detached, new_actions)
                q_new = torch.min(q1_new, q2_new).squeeze(-1)

                actor_loss = (self.alpha.detach() * log_probs - q_new).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.actor_optimizer.step()
            finally:
                self._set_critic_grad_enabled(True)

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

    def get_avg_loss(self, reset=True):
        """Get average training losses since last reset.

        Args:
            reset: if True (default), reset accumulators after reading.
        """
        if self.loss_count == 0:
            return 0.0, 0.0
        avg_critic = self.total_critic_loss / self.loss_count
        avg_actor = self.total_actor_loss / self.loss_count
        if reset:
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
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'train_step_count': self.train_step_count,
            'reward_normalizer': self.reward_normalizer.state_dict(),
            'replay_buffer': self.replay_buffer.state_dict(),
        }, path)

    def _move_optimizer_state_to_device(self, optimizer):
        """Move optimizer internal state tensors to current device."""
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def load(self, path):
        """Load agent state. Backward-compatible with old combined critic_optimizer checkpoints."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self._move_optimizer_state_to_device(self.actor_optimizer)
        # Support both new split-optimizer checkpoints and legacy combined ones
        if 'critic1_optimizer' in checkpoint and 'critic2_optimizer' in checkpoint:
            self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
            self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
            self._move_optimizer_state_to_device(self.critic1_optimizer)
            self._move_optimizer_state_to_device(self.critic2_optimizer)
        # else: legacy checkpoint has combined optimizer — skip loading it;
        #       optimizers will restart from default state (acceptable on resume)
        self.log_alpha = checkpoint['log_alpha'].to(self.device).requires_grad_(True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_optimizer.defaults['lr'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        self._move_optimizer_state_to_device(self.alpha_optimizer)
        self.train_step_count = checkpoint.get('train_step_count', 0)
        if 'reward_normalizer' in checkpoint:
            self.reward_normalizer.load_state_dict(checkpoint['reward_normalizer'])
        if 'replay_buffer' in checkpoint:
            self.replay_buffer.load_state_dict(checkpoint['replay_buffer'])
