import torch
import numpy as np

# Let's test if np.log(2) subtracts correctly
x_t = torch.randn(2, 65)
term = 2 * (np.log(2) - x_t - torch.nn.functional.softplus(-2 * x_t))
print(f"Correction shape: {term.shape}")

# Let's test freezing critics and backpropagating to inputs
class DummyCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 1)
    def forward(self, x):
        return self.fc(x)

actor = torch.nn.Linear(5, 10)
critic = DummyCritic()

x = torch.randn(1, 5)
a = actor(x)

for p in critic.parameters():
    p.requires_grad = False

q = critic(a)
q.backward()
print(f"Actor weight grad is not None: {actor.weight.grad is not None}")
print(f"Critic weight grad is None: {critic.fc.weight.grad is None}")
