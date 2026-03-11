import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from guided_diffusion.sac_agent import SACAgent

agent = SACAgent(device=torch.device("cpu"))
pred_xstart = torch.randn(1, 1, 64, 64)
adjoint_grad = torch.randn(1, 1, 64, 64)
action = agent.select_action(pred_xstart, adjoint_grad, 0.5)

modified = agent.apply_action(pred_xstart, action, adjoint_grad)
for _ in range(300):
   agent.store_transition(pred_xstart, adjoint_grad, 0.5, action, 0.1, pred_xstart, adjoint_grad, 0.4, False)
res = agent.train_step()
assert res is not None
assert "critic_loss" in res and "actor_loss" in res
print(res)
