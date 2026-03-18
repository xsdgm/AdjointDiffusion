"""Test script to verify SAC fixes work correctly (CPU only)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU

import torch
from guided_diffusion.sac_agent import SACAgent, RewardNormalizer
from guided_diffusion.gaussian_diffusion import finalize_sac_pending_transition


def test_reward_normalizer_statistics():
    normalizer = RewardNormalizer()
    rewards = [1.0, 2.0, 3.0, 4.0]

    for reward in rewards:
        normalizer.update(reward)

    assert abs(normalizer.mean - 2.5) < 1e-6, f"Unexpected mean: {normalizer.mean}"
    assert abs(normalizer.var - 1.25) < 1e-6, f"Unexpected variance: {normalizer.var}"
    normalized = normalizer.normalize_tensor(torch.tensor(rewards, dtype=torch.float32))
    expected = torch.tensor([-1.3416408, -0.4472136, 0.4472136, 1.3416408])
    assert torch.allclose(normalized, expected, atol=1e-5), f"Unexpected normalized rewards: {normalized}"
    print("RewardNormalizer statistics OK")


def test_replay_buffer_stores_raw_rewards():
    agent = SACAgent(image_size=64, patch_size=8, delta=0.1, device=torch.device("cpu"))
    pred = torch.randn(1, 1, 64, 64)
    adj = torch.randn(1, 1, 64, 64)
    action = agent.select_action(pred, adj, 0.5)

    agent.store_transition(pred, adj, 0.5, action, 3.5, pred, adj, 0.4, False)
    stored_reward = agent.replay_buffer.buffer[-1][4]
    assert stored_reward == 3.5, f"Replay buffer should store raw reward, got {stored_reward}"
    print("Replay buffer raw reward storage OK")


def test_basic():
    agent = SACAgent(image_size=64, patch_size=8, delta=0.1, device=torch.device("cpu"))
    print(f"SACAgent created OK: action_dim={agent.action_dim}, num_patches={agent.num_patches}")

    pred_xstart = torch.randn(1, 1, 64, 64)
    adjoint_grad = torch.randn(1, 1, 64, 64)

    action = agent.select_action(pred_xstart, adjoint_grad, 0.5)
    assert action.shape == (agent.action_dim,), f"Expected ({agent.action_dim},) got {action.shape}"
    print(f"select_action OK: shape={action.shape}")

    modified = agent.apply_action(pred_xstart, action, adjoint_grad)
    assert modified.shape == pred_xstart.shape
    print(f"apply_action OK: shape={modified.shape}")


def test_batched_action_path():
    agent = SACAgent(image_size=64, patch_size=8, delta=0.1, device=torch.device("cpu"))

    pred_xstart = torch.randn(2, 1, 64, 64)
    adjoint_grad = torch.randn(2, 1, 64, 64)

    action = agent.select_action(pred_xstart, adjoint_grad, 0.5)
    assert action.shape == (2, agent.action_dim), f"Expected (2, {agent.action_dim}) got {action.shape}"

    modified = agent.apply_action(pred_xstart, action, adjoint_grad)
    assert modified.shape == pred_xstart.shape
    print(f"batched apply_action OK: action_shape={action.shape}, modified_shape={modified.shape}")


def test_train_step():
    agent = SACAgent(image_size=64, patch_size=8, delta=0.1,
                     batch_size=32, device=torch.device("cpu"))
    pred = torch.randn(1, 1, 64, 64)
    adj = torch.randn(1, 1, 64, 64)
    action = agent.select_action(pred, adj, 0.5)

    # Need at least min_buffer_size (batch_size * 4 = 128) transitions
    for _ in range(130):
        agent.store_transition(pred, adj, 0.5, action, 0.1, pred, adj, 0.4, False)

    res = agent.train_step()
    assert res is not None, "train_step should return dict after enough data"
    assert 'critic_loss' in res and 'actor_loss' in res
    assert 'monitor' in res and 'q_gap_abs_mean' in res['monitor']
    print(f"train_step OK: critic_loss={res['critic_loss']:.4f}, actor_loss={res['actor_loss']:.4f}")


def test_monitor_summary_available():
    agent = SACAgent(image_size=64, patch_size=8, delta=0.1,
                     batch_size=4, min_buffer_size=4, device=torch.device("cpu"))
    pred = torch.randn(1, 1, 64, 64)
    adj = torch.randn(1, 1, 64, 64)
    action = agent.select_action(pred, adj, 0.5)

    for _ in range(4):
        agent.store_transition(pred, adj, 0.5, action, 0.1, pred, adj, 0.4, False)

    _ = agent.train_step()
    summary = agent.get_monitor_summary()
    assert summary['steps'] >= 1
    assert 'counters' in summary and 'apply_action_clamp_ratio' in summary['counters']
    print("monitor summary OK")


def test_get_avg_loss_reset():
    agent = SACAgent(image_size=64, patch_size=8, delta=0.1,
                     batch_size=32, device=torch.device("cpu"))
    pred = torch.randn(1, 1, 64, 64)
    adj = torch.randn(1, 1, 64, 64)
    action = agent.select_action(pred, adj, 0.5)

    # Need at least min_buffer_size (batch_size * 4 = 128) transitions
    for _ in range(130):
        agent.store_transition(pred, adj, 0.5, action, 0.1, pred, adj, 0.4, False)
    agent.train_step()

    # reset=False: reading twice should give the same value
    c1, a1 = agent.get_avg_loss(reset=False)
    c2, a2 = agent.get_avg_loss(reset=False)
    assert c1 == c2 and a1 == a2, "reset=False should not change values"
    print(f"get_avg_loss(reset=False) OK: consistent ({c1:.4f}, {a1:.4f})")

    # reset=True: next call should return zeros
    c3, a3 = agent.get_avg_loss(reset=True)
    c4, a4 = agent.get_avg_loss(reset=True)
    assert c4 == 0.0 and a4 == 0.0, "After reset, loss should be 0"
    print(f"get_avg_loss(reset=True) OK: after reset=({c4:.4f}, {a4:.4f})")


def test_save_load():
    import tempfile
    agent = SACAgent(image_size=64, patch_size=8, delta=0.1, device=torch.device("cpu"))
    pred = torch.randn(1, 1, 64, 64)
    adj = torch.randn(1, 1, 64, 64)
    action = agent.select_action(pred, adj, 0.5)
    agent.store_transition(pred, adj, 0.5, action, 0.3, pred, adj, 0.4, False)
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        tmp = f.name
    try:
        agent.save(tmp)
        agent2 = SACAgent(image_size=64, patch_size=8, delta=0.1, device=torch.device("cpu"))
        agent2.load(tmp)
        assert len(agent2.replay_buffer) == 1, "Replay buffer should be restored on load"
        assert abs(agent2.reward_normalizer.mean - agent.reward_normalizer.mean) < 1e-6
        print("save/load roundtrip OK")
    finally:
        os.unlink(tmp)


def test_replay_buffer_load_normalizes_legacy_entries():
    buffer = {
        'capacity': 4,
        'buffer': [(
            torch.randn(1, 1, 64, 64),
            torch.randn(1, 1, 64, 64),
            torch.tensor(0.5),
            torch.randn(65),
            torch.tensor(0.3),
            torch.randn(1, 1, 64, 64),
            torch.randn(1, 1, 64, 64),
            torch.tensor(0.4),
            torch.tensor(False),
        )]
    }

    agent = SACAgent(image_size=64, patch_size=8, delta=0.1, device=torch.device("cpu"))
    agent.replay_buffer.load_state_dict(buffer)
    (states, adj_grads, timesteps, actions, rewards,
     next_states, next_adj_grads, next_timesteps, dones) = agent.replay_buffer.sample(1)

    for tensor in (states, adj_grads, actions, next_states, next_adj_grads):
        assert tensor.device.type == "cpu", "Loaded replay tensors must be normalized onto CPU"
    assert timesteps.dtype == torch.float32
    assert rewards.dtype == torch.float32
    assert next_timesteps.dtype == torch.float32
    assert dones.dtype == torch.float32
    print("legacy replay buffer normalization OK")


def test_actor_backward_does_not_accumulate_critic_grads():
    agent = SACAgent(
        image_size=64,
        patch_size=8,
        delta=0.1,
        batch_size=4,
        min_buffer_size=4,
        device=torch.device("cpu"),
    )
    pred = torch.randn(1, 1, 64, 64)
    adj = torch.randn(1, 1, 64, 64)
    action = agent.select_action(pred, adj, 0.5)

    for _ in range(4):
        agent.store_transition(pred, adj, 0.5, action, 0.1, pred, adj, 0.4, False)

    (states, adj_grads, timesteps, _, _, _, _, _, _) = agent.replay_buffer.sample(agent.batch_size)
    states = states.to(agent.device)
    adj_grads = adj_grads.to(agent.device)
    timesteps = timesteps.to(agent.device)

    adj_max = adj_grads.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
    state_3ch = torch.cat(
        [states, adj_grads / adj_max, timesteps.view(-1, 1, 1, 1).expand_as(states)],
        dim=1,
    )

    agent.actor_optimizer.zero_grad()
    agent.critic1_optimizer.zero_grad()
    agent.critic2_optimizer.zero_grad()

    state_3ch_detached = state_3ch.detach()
    agent._set_critic_grad_enabled(False)
    try:
        new_actions, log_probs, _ = agent.actor.sample(state_3ch_detached)
        q1_new = agent.critic1(state_3ch_detached, new_actions)
        q2_new = agent.critic2(state_3ch_detached, new_actions)
        q_new = torch.min(q1_new, q2_new).squeeze(-1)
        actor_loss = (agent.alpha.detach() * log_probs - q_new).mean()
        actor_loss.backward()
    finally:
        agent._set_critic_grad_enabled(True)

    critic1_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in agent.critic1.parameters()
    )
    critic2_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in agent.critic2.parameters()
    )
    assert not critic1_grad and not critic2_grad, "Critic grads should stay frozen during actor backward"
    print("actor backward critic-freeze OK")


def test_finalize_sac_pending_transition_preserves_done_flag():
    class FakeAgent:
        def __init__(self):
            self.store_calls = []
            self.train_calls = 0

        def store_transition(self, *args):
            self.store_calls.append(args)

        def train_step(self):
            self.train_calls += 1

    agent = FakeAgent()
    pending = {
        'state': torch.zeros(1, 1, 64, 64),
        'adjoint_grad': torch.zeros(1, 1, 64, 64),
        'timestep_norm': 0.5,
        'action': torch.zeros(65),
        'reward': 1.0,
        'done': True,
    }

    finalize_sac_pending_transition(
        agent,
        pending,
        torch.ones(1, 1, 64, 64),
        torch.ones(1, 1, 64, 64),
        0.4,
    )

    assert len(agent.store_calls) == 1, "Pending transition should be stored exactly once"
    assert agent.store_calls[0][-1] is True, "Deferred transition must preserve pending done flag"
    assert agent.train_calls == 1, "Finalizing a pending transition should trigger one train step"
    print("pending transition done propagation OK")


if __name__ == "__main__":
    test_reward_normalizer_statistics()
    test_replay_buffer_stores_raw_rewards()
    test_basic()
    test_batched_action_path()
    test_train_step()
    test_monitor_summary_available()
    test_get_avg_loss_reset()
    test_save_load()
    test_replay_buffer_load_normalizes_legacy_entries()
    test_actor_backward_does_not_accumulate_critic_grads()
    test_finalize_sac_pending_transition_preserves_done_flag()
    print("\nAll tests passed!")
