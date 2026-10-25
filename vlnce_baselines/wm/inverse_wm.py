"""Inverse Action Agent.

    a_t = G_phi(z_t, z_hat_{t+1}, I)

Goal-conditioned inverse dynamics: given the current latent state and a
target latent state (imagined by the forward agent, or the real next state in
hindsight), infer which slot of the topological map realises the transition,
regress the relative displacement it implies, and estimate whether the
low-level controller can physically execute it (obstacles, sliding, failed
try-outs in VLN-CE). Inverse supervision also regularises the latent space
towards controllable factors (direction, displacement, reachability) and
prevents representation collapse (cf. SMWM, FutureNav).
"""

import torch
import torch.nn as nn

from vlnce_baselines.wm.latent import mlp

# indices of the 7-d ETPNav pos feature regressed by the inverse agent:
# sin(rel_heading), cos(rel_heading), line_dist / MAX_DIST
POSE_DIMS = [0, 1, 4]


class InverseActionAgent(nn.Module):
    def __init__(self, latent_dim=256, dropout=0.1):
        super().__init__()
        d = latent_dim
        self.pair = mlp(d * 4, d, d, dropout)
        self.slot_score = mlp(d * 3, d, 1, dropout)
        self.pose_head = mlp(d, d, len(POSE_DIMS), dropout)
        self.exec_head = mlp(d * 3, d, 1, dropout)

    def forward(self, z_t, z_target, w, c, slot_masks):
        """z_t, z_target, c: B x d; w: B x N x d -> logits B x N, pose B x 3."""
        pair = self.pair(torch.cat([z_t, z_target, z_target - z_t, c], -1))
        N = w.size(1)
        pair_n = pair.unsqueeze(1).expand(-1, N, -1)
        logits = self.slot_score(torch.cat([pair_n, w, pair_n * w], -1)).squeeze(-1)
        logits = logits.masked_fill(~slot_masks, -float('inf'))
        pose = self.pose_head(pair)
        return logits, pose

    def executability(self, z_t, w, z_hat):
        """Logit that committing to slot k physically arrives at its ghost."""
        N = w.size(1)
        z_n = z_t.unsqueeze(1).expand(-1, N, -1)
        return self.exec_head(torch.cat([z_n, w, z_hat], -1)).squeeze(-1)
