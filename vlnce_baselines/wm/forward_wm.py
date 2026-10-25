"""Forward Imagination Agent.

    z_hat_{t+1}^k = F_theta(z_t, w_t^k, I, M_t)

An RSSM-style latent transition model (Hafner et al. 2019, as used for
VLN-CE by NavMorph) conditioned on the ETPNav waypoint prior w^k, the
instruction reading c_t and the corrective-memory read-out m_t. It imagines
the latent state reached by committing to every slot k of the topological map
in one batched step, scores how well each imagined state satisfies the
instruction (forward target), and keeps a recurrent belief (h_t, s_t) that is
corrected by the posterior once the real observation arrives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vlnce_baselines.wm.latent import mlp

MIN_STD = 0.05


class ForwardImaginationAgent(nn.Module):
    def __init__(self, latent_dim=256, deter_dim=256, stoch_dim=32,
                 feat_dim=768, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim

        self.init_h = nn.Linear(latent_dim, deter_dim)
        self.transition_in = nn.Sequential(
            nn.Linear(stoch_dim + latent_dim * 3, deter_dim), nn.GELU(),
        )
        self.cell = nn.GRUCell(deter_dim, deter_dim)
        self.prior = mlp(deter_dim, deter_dim, stoch_dim * 2, dropout)
        self.posterior = mlp(deter_dim + latent_dim, deter_dim, stoch_dim * 2, dropout)
        # latent decoder: imagined state in the same space as StateEncoder
        self.dec_latent = mlp(deter_dim + stoch_dim, deter_dim, latent_dim, dropout, norm_out=True)
        # feature decoder: reconstruct ETPNav's 768-d panorama embedding
        self.dec_feat = mlp(deter_dim + stoch_dim, deter_dim * 2, feat_dim, dropout)
        # instruction-progress value of an imagined state (forward target score)
        self.value = mlp(latent_dim * 3, latent_dim, 1, dropout)
        self.stop_value = mlp(latent_dim * 3, latent_dim, 1, dropout)

    # ------------------------------------------------------------------ belief
    @staticmethod
    def _dist(stats):
        mean, std = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(std) + MIN_STD
        return mean, std

    def init_belief(self, z0):
        h0 = torch.tanh(self.init_h(z0))
        mean, std = self._dist(self.posterior(torch.cat([h0, z0], -1)))
        return h0, mean

    def observe(self, h, z_obs, sample=True):
        """Posterior q(s | h, E(o)) after the real observation arrives."""
        mean, std = self._dist(self.posterior(torch.cat([h, z_obs], -1)))
        s = mean + std * torch.randn_like(std) if sample else mean
        return s, mean, std

    def decode_feat(self, h, s):
        return self.dec_feat(torch.cat([h, s], -1))

    def decode_latent(self, h, s):
        return self.dec_latent(torch.cat([h, s], -1))

    # ------------------------------------------------------------- imagination
    def imagine(self, h, s, z_t, w, m, c, slot_masks):
        """Imagine the next latent state for every graph slot in one shot.

        h, s: belief (B x Dh, B x Ds); z_t: current latent (B x d);
        w: waypoint priors (B x N x d), slot 0 is STOP; m: memory read-out
        (B x d); c: instruction reading (B x d); slot_masks: B x N bool,
        True for slots the agent may select.
        """
        B, N, d = w.shape
        exp = lambda x: x.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
        inp = self.transition_in(torch.cat([exp(s), w.reshape(B * N, d), exp(m), exp(c)], -1))
        h_k = self.cell(inp, exp(h))
        p_mean, p_std = self._dist(self.prior(h_k))
        z_hat = self.decode_latent(h_k, p_mean)

        h_k = h_k.view(B, N, -1)
        p_mean = p_mean.view(B, N, -1)
        p_std = p_std.view(B, N, -1)
        z_hat = z_hat.view(B, N, -1)
        # STOP slot: no transition, the target is the current state itself
        z_hat = torch.cat([z_t.unsqueeze(1), z_hat[:, 1:]], 1)

        c_n = c.unsqueeze(1).expand(-1, N, -1)
        value = self.value(torch.cat([z_hat, c_n, z_hat * c_n], -1)).squeeze(-1)
        stop_value = self.stop_value(torch.cat([z_t, c, m], -1))
        value = torch.cat([stop_value, value[:, 1:]], 1)
        value = value.masked_fill(~slot_masks, -float('inf'))

        return {
            'h_k': h_k,              # B x N x Dh, deterministic state per slot
            'prior_mean': p_mean,    # B x N x Ds
            'prior_std': p_std,      # B x N x Ds
            'z_hat': z_hat,          # B x N x d, imagined next latent state
            'value': value,          # B x N, forward target logits
            'uncertainty': p_std.mean(-1),  # B x N, aleatoric spread
        }

    def state_value(self, z, c):
        return self.value(torch.cat([z, c, z * c], -1)).squeeze(-1)


def gaussian_kl(q_mean, q_std, p_mean, p_std):
    var_ratio = (q_std / p_std) ** 2
    t1 = ((q_mean - p_mean) / p_std) ** 2
    return 0.5 * (var_ratio + t1 - 1 - torch.log(var_ratio)).sum(-1)
