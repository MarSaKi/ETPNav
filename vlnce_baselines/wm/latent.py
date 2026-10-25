import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim, hidden, out_dim, dropout=0.0, norm_out=False):
    layers = [
        nn.Linear(in_dim, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    ]
    if norm_out:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


class StateEncoder(nn.Module):
    """z_t = E(o_t): projects ETPNav's frozen 768-d panorama embedding
    (masked average of the panorama encoder output at the current node)
    into the world-model latent space."""

    def __init__(self, feat_dim=768, latent_dim=256, dropout=0.1):
        super().__init__()
        self.net = mlp(feat_dim, latent_dim * 2, latent_dim, dropout, norm_out=True)

    def forward(self, pano_feat):
        return self.net(pano_feat)


class WaypointEncoder(nn.Module):
    """w_k: action / waypoint prior of graph slot k.

    Built from what ETPNav already computes for every slot of the topological
    map: the aggregated view feature of the ghost (gmap_img_fts), the
    instruction-aware node embedding of the frozen planner (gmap_embeds) and
    the 7-d relative pose feature (heading, elevation, line / shortest-path
    distance, steps). Slot 0 is ETPNav's STOP slot and gets a learned
    embedding.
    """

    def __init__(self, feat_dim=768, pos_dim=7, latent_dim=256, dropout=0.1):
        super().__init__()
        self.img_proj = nn.Linear(feat_dim, latent_dim)
        self.ctx_proj = nn.Linear(feat_dim, latent_dim)
        self.pos_proj = nn.Linear(pos_dim, latent_dim)
        self.out = nn.Sequential(nn.LayerNorm(latent_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(latent_dim, latent_dim))
        self.stop_embed = nn.Parameter(torch.zeros(latent_dim))
        nn.init.normal_(self.stop_embed, std=0.02)

    def forward(self, gmap_img_fts, gmap_embeds, gmap_pos_fts):
        w = self.img_proj(gmap_img_fts) + self.ctx_proj(gmap_embeds) + \
            self.pos_proj(gmap_pos_fts)
        w = self.out(w)
        stop = self.stop_embed.view(1, 1, -1).expand(w.size(0), 1, -1)
        return torch.cat([stop, w[:, 1:]], dim=1)


class InstructionReader(nn.Module):
    """c_t = Attn(q(h_t, z_t), I): state-dependent reading of the instruction,
    so the world model tracks which sub-instruction is being executed."""

    def __init__(self, txt_dim=768, query_dim=512, latent_dim=256):
        super().__init__()
        self.q = nn.Linear(query_dim, latent_dim)
        self.k = nn.Linear(txt_dim, latent_dim)
        self.v = nn.Linear(txt_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        self.scale = 1.0 / math.sqrt(latent_dim)

    def forward(self, query, txt_embeds, txt_masks):
        q = self.q(query).unsqueeze(1)                       # B x 1 x d
        k = self.k(txt_embeds)                               # B x L x d
        v = self.v(txt_embeds)
        att = torch.matmul(q, k.transpose(1, 2)).squeeze(1) * self.scale
        att = att.masked_fill(~txt_masks.bool(), -1e4)
        att = F.softmax(att, dim=-1)
        ctx = torch.matmul(att.unsqueeze(1), v).squeeze(1)
        return self.norm(ctx), att
