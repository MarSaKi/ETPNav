"""Uncertainty-aware fusion of the frozen ETPNav planner with the world models.

    logits = l_ETP + lam_F * log pi_F + lam_G * log pi_G + lam_E * log p_exec
             + gamma * rho_t * log pi_C

lam_* are produced per step from ETPNav's own confidence (entropy, max prob)
and the previous deviation estimate, so the world models take over more
authority where the reactive planner is uncertain. rho_t is the revision
strength issued by the closed-loop runtime (0 when the memory agent says
"continue").
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_GATE_FEATS = 3
NUM_SOURCES = 3


def masked_log_softmax(logits, valid):
    logits = logits.masked_fill(~valid, -float('inf'))
    return F.log_softmax(logits, -1)


def etp_confidence_feats(etp_logits, valid):
    """Normalised entropy and max probability of ETPNav's decision (B x 2)."""
    logp = masked_log_softmax(etp_logits, valid)
    p = logp.exp()
    ent = -(p * logp.masked_fill(~valid, 0.)).sum(-1)
    n = valid.float().sum(-1).clamp(min=2.)
    return torch.stack([ent / torch.log(n), p.max(-1)[0]], -1)


class FusionGate(nn.Module):
    def __init__(self, init_weight=0.3, init_gamma=1.0, hidden=32):
        super().__init__()
        inv_sp = lambda x: float(torch.log(torch.expm1(torch.tensor(x))))
        self.base = nn.Parameter(torch.full((NUM_SOURCES,), inv_sp(init_weight)))
        self.gamma = nn.Parameter(torch.tensor(inv_sp(init_gamma)))
        self.gate = nn.Sequential(
            nn.Linear(NUM_GATE_FEATS, hidden), nn.GELU(), nn.Linear(hidden, NUM_SOURCES),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def weights(self, gate_feats):
        return F.softplus(self.base.unsqueeze(0) + self.gate(gate_feats))

    def forward(self, etp_logits, fwd_logits, inv_logits, exec_logits,
                corr_logits, revise_strength, prev_dev, valid,
                use=(True, True, True, True)):
        """All slot tensors are B x N; revise_strength, prev_dev: B.
        `use` switches (forward, inverse, executability, correction) for
        ablations."""
        conf = etp_confidence_feats(etp_logits, valid)
        lam = self.weights(torch.cat([conf, prev_dev.unsqueeze(-1)], -1))

        # zero the masked slots *before* scaling: lam * (-inf) would turn
        # into NaN gradients for lam even though those slots are dropped
        safe = lambda x: x.masked_fill(~valid, 0.)
        fused = safe(etp_logits)
        if use[0]:
            fused = fused + lam[:, 0:1] * safe(masked_log_softmax(fwd_logits, valid))
        if use[1]:
            fused = fused + lam[:, 1:2] * safe(masked_log_softmax(inv_logits, valid))
        if use[2]:
            fused = fused + lam[:, 2:3] * safe(F.logsigmoid(exec_logits))
        if use[3]:
            gamma = F.softplus(self.gamma)
            fused = fused + gamma * revise_strength.unsqueeze(-1) * \
                safe(masked_log_softmax(corr_logits, valid))
        return fused.masked_fill(~valid, -float('inf')), lam
