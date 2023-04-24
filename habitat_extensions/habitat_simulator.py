#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Union,
    cast,
)

import numpy as np
from gym import spaces
from gym.spaces.box import Box
from numpy import ndarray

if TYPE_CHECKING:
    from torch import Tensor

import habitat_sim

from habitat_sim.simulator import MutableMapping, MutableMapping_T
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    Config,
    DepthSensor,
    Observations,
    RGBSensor,
    SemanticSensor,
    Sensor,
    SensorSuite,
    ShortestPathPoint,
    Simulator,
    VisualObservation,
)
from habitat.core.spaces import Space

# inherit habitat-lab/habitat/sims/habitat_simulator/habitat_simulator.py
@registry.register_simulator(name="Sim-v1")
class Simulator(HabitatSim):
    r"""Simulator wrapper over habitat-sim

    habitat-sim repo: https://github.com/facebookresearch/habitat-sim

    Args:
        config: configuration for initializing the simulator.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def step_without_obs(self,
        action: Union[str, int, MutableMapping_T[int, Union[str, int]]],
        dt: float = 1.0 / 60.0,):
        self._num_total_frames += 1
        if isinstance(action, MutableMapping):
            return_single = False
        else:
            action = cast(Dict[int, Union[str, int]], {self._default_agent_id: action})
            return_single = True
        collided_dict: Dict[int, bool] = {}
        for agent_id, agent_act in action.items():
            agent = self.get_agent(agent_id)
            collided_dict[agent_id] = agent.act(agent_act)
            self.__last_state[agent_id] = agent.get_state()

        # # step physics by dt
        # step_start_Time = time.time()
        # super().step_world(dt)
        # self._previous_step_time = time.time() - step_start_Time

        multi_observations = {}
        for agent_id in action.keys():
            agent_observation = {}
            agent_observation["collided"] = collided_dict[agent_id]
            multi_observations[agent_id] = agent_observation

        if return_single:
            sim_obs = multi_observations[self._default_agent_id]
        else:
            sim_obs = multi_observations

        self._prev_sim_obs = sim_obs
