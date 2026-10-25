"""Bidirectional world models + corrective memory for ETPNav (IEVC loop).

Imagine (forward WM) -> Execute (inverse WM picks the action) ->
Verify (memory agent compares prediction with observation) ->
Correct (memory agent revises the next target / vetoes a wrong stop).

All modules are plug-ins on top of a frozen ETPNav backbone; see
docs/WM_DESIGN.md for the full description.
"""

from vlnce_baselines.wm.latent import StateEncoder, InstructionReader
from vlnce_baselines.wm.forward_wm import ForwardImaginationAgent
from vlnce_baselines.wm.inverse_wm import InverseActionAgent
from vlnce_baselines.wm.memory_agent import (
    CorrectiveMemory, MemoryAgent, NUM_VERIFY_FEATS,
)
from vlnce_baselines.wm.fusion import FusionGate
from vlnce_baselines.wm.bidir_wm import BiWorldModelNav
from vlnce_baselines.wm.runtime import ClosedLoopRuntime, EpisodeState

__all__ = [
    'StateEncoder', 'InstructionReader', 'ForwardImaginationAgent',
    'InverseActionAgent', 'CorrectiveMemory', 'MemoryAgent',
    'NUM_VERIFY_FEATS', 'FusionGate', 'BiWorldModelNav',
    'ClosedLoopRuntime', 'EpisodeState',
]
