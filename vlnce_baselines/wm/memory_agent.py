"""Memory Agent: Verify & Correct.

    M_{t+1} = U_psi(M_t, z_hat_{t+1}, z_{t+1}, a_t, I)

Three parts:

* CorrectiveMemory -- an episodic slot memory keyed by topological-map nodes.
  Writes are gated by the verified reliability of the transition (only
  observation-verified experience is stored at full strength), a revisit of a
  node refreshes its slot instead of appending a duplicate, and unreliable
  writes are down-weighted at read time (log-reliability attention bias).
  These three properties follow the memory-extraction / freshness /
  interference-rejection axes of WorldMemArena.
* Verifier -- compares the executed prediction with the real observation
  (prediction gap, inverse consistency, pose error, progress change, history
  statistics) and outputs the probability that the agent has deviated from
  the instructed route, as opposed to a benign surprise caused by partial
  observability.
* Corrector -- when a deviation is detected, scores every selectable slot as
  a revised future target using the memory, the node verification flags and
  the alternatives imagined earlier at verified nodes. The resulting bias is
  fed back into the next forward / inverse decision; executed actions are
  never rewritten.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vlnce_baselines.wm.latent import mlp

VERIFY_FEATS = [
    'pred_gap_cos',      # 1 - cos(z_hat, z_obs)
    'pred_nll',          # Gaussian NLL of the posterior mean under the prior
    'post_prior_kl',     # KL(q || p), per stochastic dim
    'inverse_nll',       # -log G(a_t | z_t, z_obs): hindsight action consistency
    'pose_error',        # |arrived position - ghost position| / 3 m
    'value_drop',        # tanh(v(z_hat) - v(z_obs)): expected vs realised progress
    'pred_uncertainty',  # aleatoric spread of the prior for the executed slot
    'memory_similarity', # max cosine similarity to stored states (loops)
    'gap_zscore',        # prediction gap relative to this episode's history
    'etp_stop_prob',     # ETPNav stop probability at the new node
    'etp_entropy',       # normalised entropy of ETPNav's action distribution
    'step_fraction',     # t / T_max
    'suspect_streak',    # consecutive deviation flags / 3
    'exec_prob',         # executability predicted before committing
]
NUM_VERIFY_FEATS = len(VERIFY_FEATS)

SLOT_FLAGS = [
    'front_suspect_frac',   # fraction of the slot's front nodes flagged as deviated
    'front_verified_frac',  # fraction of front nodes verified by observation
    'stored_alt_value',     # imagined value recorded when the front node was visited
]
NUM_SLOT_FLAGS = len(SLOT_FLAGS)


class CorrectiveMemory(nn.Module):
    def __init__(self, latent_dim=256, dropout=0.1):
        super().__init__()
        d = latent_dim
        self.write_in = mlp(d * 4 + NUM_VERIFY_FEATS, d, d, dropout)
        self.write_cell = nn.GRUCell(d, d)
        self.query = nn.Linear(d * 2, d)
        self.key = nn.Linear(d, d)
        self.null_key = nn.Parameter(torch.zeros(d))
        self.null_value = nn.Parameter(torch.zeros(d))
        self.out = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d))
        self.scale = 1.0 / math.sqrt(d)
        nn.init.normal_(self.null_key, std=0.02)

    def write(self, prev_value, z_hat, z_obs, w_a, c, verify_feats):
        """U_psi for one slot. prev_value: the slot's previous content (zeros
        for a new node). The caller blends the result with prev_value using
        the reliability gate."""
        x = self.write_in(torch.cat([z_hat, z_obs, w_a, c, verify_feats], -1))
        return self.write_cell(x, prev_value)

    def read(self, z_t, c, keys, values, log_rel, slot_mask):
        """keys/values: B x S x d, log_rel: B x S, slot_mask: B x S bool.
        A learned null slot keeps the read defined for an empty memory."""
        B = z_t.size(0)
        q = self.query(torch.cat([z_t, c], -1))
        nk = self.null_key.view(1, 1, -1).expand(B, 1, -1)
        nv = self.null_value.view(1, 1, -1).expand(B, 1, -1)
        k = self.key(torch.cat([nk, keys], 1))
        v = torch.cat([nv, values], 1)
        logits = torch.matmul(k, q.unsqueeze(-1)).squeeze(-1) * self.scale
        bias = torch.cat([log_rel.new_zeros(B, 1), log_rel], 1)
        mask = torch.cat([slot_mask.new_ones(B, 1), slot_mask], 1)
        logits = (logits + bias).masked_fill(~mask, -1e4)
        att = F.softmax(logits, -1)
        return self.out(torch.matmul(att.unsqueeze(1), v).squeeze(1)), att


class MemoryAgent(nn.Module):
    def __init__(self, latent_dim=256, dropout=0.1):
        super().__init__()
        d = latent_dim
        self.memory = CorrectiveMemory(latent_dim, dropout)
        self.feat_norm = nn.LayerNorm(NUM_VERIFY_FEATS)
        self.verifier = mlp(NUM_VERIFY_FEATS + d * 2, d, 1, dropout)
        self.corrector = mlp(d * 3 + NUM_SLOT_FLAGS, d, 1, dropout)

    def verify(self, verify_feats, z_hat, z_obs):
        """Deviation logit for the transition just executed (B,)."""
        f = self.feat_norm(verify_feats)
        gap = torch.abs(z_hat - z_obs)
        return self.verifier(torch.cat([f, gap, z_obs], -1)).squeeze(-1)

    def correct(self, w, z_hat, m, slot_flags, slot_masks):
        """Revised-future-target logits over graph slots (B x N)."""
        N = w.size(1)
        m_n = m.unsqueeze(1).expand(-1, N, -1)
        logits = self.corrector(torch.cat([w, z_hat, m_n, slot_flags], -1)).squeeze(-1)
        return logits.masked_fill(~slot_masks, -float('inf'))
