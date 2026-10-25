import torch
import torch.nn as nn

from vlnce_baselines.wm.latent import StateEncoder, WaypointEncoder, InstructionReader
from vlnce_baselines.wm.forward_wm import ForwardImaginationAgent
from vlnce_baselines.wm.inverse_wm import InverseActionAgent
from vlnce_baselines.wm.memory_agent import MemoryAgent
from vlnce_baselines.wm.fusion import FusionGate


class BiWorldModelNav(nn.Module):
    """Container of all trainable parameters of the closed loop. The ETPNav
    backbone stays outside (frozen); this module only consumes tensors that
    ETPNav already produces at every step."""

    def __init__(self, feat_dim=768, latent_dim=256, deter_dim=256,
                 stoch_dim=32, dropout=0.1, fusion_init=0.3, gamma_init=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.state_enc = StateEncoder(feat_dim, latent_dim, dropout)
        self.waypoint_enc = WaypointEncoder(feat_dim, 7, latent_dim, dropout)
        self.reader = InstructionReader(feat_dim, deter_dim + latent_dim, latent_dim)
        self.forward_wm = ForwardImaginationAgent(latent_dim, deter_dim, stoch_dim,
                                                  feat_dim, dropout)
        self.inverse_wm = InverseActionAgent(latent_dim, dropout)
        self.memory_agent = MemoryAgent(latent_dim, dropout)
        self.fusion = FusionGate(fusion_init, gamma_init)

    @classmethod
    def from_config(cls, wm_cfg):
        return cls(
            feat_dim=768,
            latent_dim=wm_cfg.latent_dim,
            deter_dim=wm_cfg.deter_dim,
            stoch_dim=wm_cfg.stoch_dim,
            dropout=wm_cfg.dropout,
            fusion_init=wm_cfg.fusion_init,
            gamma_init=wm_cfg.correction_gamma,
        )
