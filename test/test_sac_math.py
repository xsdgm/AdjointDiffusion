import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

mean = torch.tensor([[-0.5, 0.0, 0.5]])
std = torch.tensor([[0.1, 0.1, 0.1]])
normal = Normal(mean, std)
x_t = mean
action = torch.tanh(x_t)

# Their formula
log_prob1 = normal.log_prob(x_t)
log_prob1 -= (2 * (np.log(2) - x_t - F.softplus(-2 * x_t)))

# Standard formula
log_prob2 = normal.log_prob(x_t)
log_prob2 -= torch.log(1 - action.pow(2) + 1e-6)

print("Formula 1:", log_prob1)
print("Formula 2:", log_prob2)
assert torch.allclose(log_prob1, log_prob2, atol=1e-6)
