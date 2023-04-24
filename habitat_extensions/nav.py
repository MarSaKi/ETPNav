from turtle import heading
from typing import Any, List, Optional, Tuple

import math
import numpy as np
import random

from habitat.core.embodied_task import (
    SimulatorTaskAction,
)
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector

# @registry.register_task_action
# class MoveForwardByDistanceAction(SimulatorTaskAction):
#     def step(self, *args: Any, distance: float,**kwargs: Any):
#         r"""Update ``_metric``, this method is called from ``Env`` on each
#         ``step``.
#         """
#         original_amount = self._sim.get_agent(0).agent_config.action_space[1].actuation.amount
#         self._sim.get_agent(0).agent_config.action_space[1].actuation.amount = distance
#         output = self._sim.step(HabitatSimActions.MOVE_FORWARD)
#         self._sim.get_agent(0).agent_config.action_space[1].actuation.amount = original_amount
#         return output


@registry.register_task_action
class MoveHighToLowAction(SimulatorTaskAction):
    def cal_heading(agent_state):
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1])
        )
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return math.degrees(heading)

    def turn(self, angle):
        ''' angle: 0 ~ 360 degree '''
        left_action = HabitatSimActions.TURN_LEFT
        right_action = HabitatSimActions.TURN_RIGHT
        turn_unit = self._sim.get_agent(0).agent_config.action_space[left_action].actuation.amount
        angle = round(angle / turn_unit) * turn_unit

        if 180 < angle <= 360:
                angle -= 360
        if angle >=0:
            turn_actions = [left_action] * (angle // turn_unit)
        else:
            turn_actions = [right_action] * (-angle // turn_unit)
        # print(angle)
        # print(self.cal_heading(self._sim.get_agent_state()))
        for turn_action in turn_actions:
            self._sim.step_without_obs(turn_action)
            # print(self.cal_heading(self._sim.get_agent_state()))
    
    def step(self, *args: Any, 
            angle: float, distance: float,
            niu1niu: bool = False,
            **kwargs: Any):
        r"""This control method is called from ``Env`` on each ``step``.
        """
        if not niu1niu:
            init_state = self._sim.get_agent_state()

            forward_action = HabitatSimActions.MOVE_FORWARD

            init_forward = self._sim.get_agent(0).agent_config.action_space[
                forward_action].actuation.amount
            theta = np.arctan2(init_state.rotation.imag[1], init_state.rotation.real) + angle / 2
            rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
            self._sim.set_agent_state(init_state.position, rotation)

            ksteps = int(distance//init_forward)
            for k in range(ksteps):
                if k == ksteps - 1:
                    output = self._sim.step(forward_action)
                else:
                    self._sim.step_without_obs(forward_action)
            
            return output
        
        elif niu1niu:
            positions = []
            collisions = []
            forward_action = HabitatSimActions.MOVE_FORWARD
            left_action = HabitatSimActions.TURN_LEFT
            right_action = HabitatSimActions.TURN_RIGHT
            foward_unit = self._sim.get_agent(0).agent_config.action_space[forward_action].actuation.amount

            angle = math.degrees(angle)
            self.turn(angle)

            forward_step = int(distance // foward_unit)
            for k in range(forward_step):
                if k == forward_step - 1:
                    output = self._sim.step(forward_action)
                    positions.append(self._sim.get_agent_state().position)
                    collisions.append(self._sim.previous_step_collided)
                    break
                else:
                    self._sim.step_without_obs(forward_action)
                    positions.append(self._sim.get_agent_state().position)
                    collisions.append(self._sim.previous_step_collided)
                    if self._sim.previous_step_collided:
                        output = self._sim.step(forward_action)
                        break
            
            # left forward step
            forward_step = forward_step - len(collisions)
            if forward_step > 0:
                # assert self._sim.previous_step_collided == True
                init_try_angle = random.choice([90, 270]) # left or right randomly
                self.turn(init_try_angle)

                if init_try_angle == 90: # from left to right
                    turn_seqs = [
                        {'head_turns': [],               'tail_turns': [right_action]*3 }, # 90
                        {'head_turns': [right_action],   'tail_turns': [right_action]*2 }, # 60
                        {'head_turns': [right_action],   'tail_turns': [right_action] }, # 30
                        {'head_turns': [right_action]*2, 'tail_turns': [left_action] }, # -30
                        {'head_turns': [right_action],   'tail_turns': [left_action]*2 }, # -60
                        {'head_turns': [right_action],   'tail_turns': [left_action]*3 }, # -90
                    ]
                elif init_try_angle == 270: # from right to left
                    turn_seqs = [
                        {'head_turns': [],              'tail_turns': [left_action]*3 }, # -90
                        {'head_turns': [left_action],   'tail_turns': [left_action]*2 }, # -60
                        {'head_turns': [left_action],   'tail_turns': [left_action] }, # -30
                        {'head_turns': [left_action]*2, 'tail_turns': [right_action] }, # 30
                        {'head_turns': [left_action],   'tail_turns': [right_action]*2 }, # 60
                        {'head_turns': [left_action],   'tail_turns': [right_action]*3 }, # 90
                    ]
                # try each direction, if pos change, do tail_turns, then do left forward actions
                for turn_seq in turn_seqs:
                    for turn in turn_seq['head_turns']:
                        self._sim.step_without_obs(turn)
                    prev_position = self._sim.get_agent_state().position
                    self._sim.step_without_obs(forward_action)
                    post_posiiton = self._sim.get_agent_state().position
                    # pos change
                    if list(prev_position) != list(post_posiiton):
                        positions.append(self._sim.get_agent_state().position)
                        collisions.append(self._sim.previous_step_collided)
                        # do tail_turns
                        for turn in turn_seq['tail_turns']:
                            self._sim.step_without_obs(turn)
                        # do left forward actions
                        for k in range(forward_step):
                            if k == forward_step - 1:
                                output = self._sim.step(forward_action)
                                positions.append(self._sim.get_agent_state().position)
                                collisions.append(self._sim.previous_step_collided)
                            else:
                                self._sim.step_without_obs(forward_action)
                                positions.append(self._sim.get_agent_state().position)
                                collisions.append(self._sim.previous_step_collided)
                                if self._sim.previous_step_collided:
                                    output = self._sim.step(forward_action)
                                    break
                        break

            return output


@registry.register_task_action
class MoveHighToLowActionEval(SimulatorTaskAction):
    @staticmethod
    def cal_heading(agent_state):
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1])
        )
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return math.degrees(heading)

    def turn(self, angle):
        ''' angle: 0 ~ 360 degree '''
        left_action = HabitatSimActions.TURN_LEFT
        right_action = HabitatSimActions.TURN_RIGHT
        turn_unit = self._sim.get_agent(0).agent_config.action_space[left_action].actuation.amount
        angle = round(angle / turn_unit) * turn_unit

        if 180 < angle <= 360:
                angle -= 360
        if angle >=0:
            turn_actions = [left_action] * (angle // turn_unit)
        else:
            turn_actions = [right_action] * (-angle // turn_unit)
        # print(angle)
        # print(self.cal_heading(self._sim.get_agent_state()))
        for turn_action in turn_actions:
            self._sim.step_without_obs(turn_action)
            # print(self.cal_heading(self._sim.get_agent_state()))

    def step(self, *args: Any, 
            angle: float, distance: float,
            niu1niu: bool = False,
            **kwargs: Any):
        r"""This control method is called from ``Env`` on each ``step``.
        """
        if not niu1niu:
            init_state = self._sim.get_agent_state()

            positions = []
            collisions = []
            forward_action = HabitatSimActions.MOVE_FORWARD

            init_forward = self._sim.get_agent(0).agent_config.action_space[
                forward_action].actuation.amount
            theta = np.arctan2(init_state.rotation.imag[1], 
                init_state.rotation.real) + angle / 2
            rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
            self._sim.set_agent_state(init_state.position, rotation)

            ksteps = int(distance//init_forward)
            for k in range(ksteps):
                if k == ksteps - 1:
                    output = self._sim.step(forward_action)
                else:
                    self._sim.step_without_obs(forward_action)
                positions.append(self._sim.get_agent_state().position)
                collisions.append(self._sim.previous_step_collided)

            output['positions'] = positions
            output['collisions'] = collisions

            return output

        elif niu1niu:
            positions = []
            collisions = []
            forward_action = HabitatSimActions.MOVE_FORWARD
            left_action = HabitatSimActions.TURN_LEFT
            right_action = HabitatSimActions.TURN_RIGHT
            foward_unit = self._sim.get_agent(0).agent_config.action_space[forward_action].actuation.amount

            angle = math.degrees(angle)
            self.turn(angle)

            forward_step = int(distance // foward_unit)
            for k in range(forward_step):
                if k == forward_step - 1:
                    output = self._sim.step(forward_action)
                    positions.append(self._sim.get_agent_state().position)
                    collisions.append(self._sim.previous_step_collided)
                    break
                else:
                    self._sim.step_without_obs(forward_action)
                    positions.append(self._sim.get_agent_state().position)
                    collisions.append(self._sim.previous_step_collided)
                    if self._sim.previous_step_collided:
                        output = self._sim.step(forward_action)
                        break
            
            # left forward step
            forward_step = forward_step - len(collisions)
            if forward_step > 0:
                # assert self._sim.previous_step_collided == True
                init_try_angle = random.choice([90, 270]) # left or right randomly
                self.turn(init_try_angle)

                if init_try_angle == 90: # from left to right
                    turn_seqs = [
                        {'head_turns': [],               'tail_turns': [right_action]*3 }, # 90
                        {'head_turns': [right_action],   'tail_turns': [right_action]*2 }, # 60
                        {'head_turns': [right_action],   'tail_turns': [right_action] }, # 30
                        {'head_turns': [right_action]*2, 'tail_turns': [left_action] }, # -30
                        {'head_turns': [right_action],   'tail_turns': [left_action]*2 }, # -60
                        {'head_turns': [right_action],   'tail_turns': [left_action]*3 }, # -90
                    ]
                elif init_try_angle == 270: # from right to left
                    turn_seqs = [
                        {'head_turns': [],              'tail_turns': [left_action]*3 }, # -90
                        {'head_turns': [left_action],   'tail_turns': [left_action]*2 }, # -60
                        {'head_turns': [left_action],   'tail_turns': [left_action] }, # -30
                        {'head_turns': [left_action]*2, 'tail_turns': [right_action] }, # 30
                        {'head_turns': [left_action],   'tail_turns': [right_action]*2 }, # 60
                        {'head_turns': [left_action],   'tail_turns': [right_action]*3 }, # 90
                    ]
                # try each direction, if pos change, do tail_turns, then do left forward actions
                for turn_seq in turn_seqs:
                    for turn in turn_seq['head_turns']:
                        self._sim.step_without_obs(turn)
                    prev_position = self._sim.get_agent_state().position
                    self._sim.step_without_obs(forward_action)
                    post_posiiton = self._sim.get_agent_state().position
                    # pos change
                    if list(prev_position) != list(post_posiiton):
                        positions.append(self._sim.get_agent_state().position)
                        collisions.append(self._sim.previous_step_collided)
                        # do tail_turns
                        for turn in turn_seq['tail_turns']:
                            self._sim.step_without_obs(turn)
                        # do left forward actions
                        for k in range(forward_step):
                            if k == forward_step - 1:
                                output = self._sim.step(forward_action)
                                positions.append(self._sim.get_agent_state().position)
                                collisions.append(self._sim.previous_step_collided)
                            else:
                                self._sim.step_without_obs(forward_action)
                                positions.append(self._sim.get_agent_state().position)
                                collisions.append(self._sim.previous_step_collided)
                                if self._sim.previous_step_collided:
                                    output = self._sim.step(forward_action)
                                    break
                        break

            output['positions'] = positions
            output['collisions'] = collisions

            return output

@registry.register_task_action
class MoveHighToLowActionInference(SimulatorTaskAction):
    def turn(self, angle):
        ''' angle: 0 ~ 360 degree '''
        left_action = HabitatSimActions.TURN_LEFT
        right_action = HabitatSimActions.TURN_RIGHT
        turn_unit = self._sim.get_agent(0).agent_config.action_space[left_action].actuation.amount
        angle = round(angle / turn_unit) * turn_unit

        if 180 < angle <= 360:
                angle -= 360
        if angle >=0:
            turn_actions = [left_action] * (angle // turn_unit)
        else:
            turn_actions = [right_action] * (-angle // turn_unit)
        for turn_action in turn_actions:
            self._sim.step_without_obs(turn_action)
    
    def get_agent_info(self):
        agent_state = self._sim.get_agent_state()
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1])
        )
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return {
            "position": agent_state.position.tolist(),
            "heading": heading,
            "stop": False,
        }

    def step(self, *args: Any, 
            angle: float, distance: float,
            niu1niu: bool = False,
            **kwargs: Any):
        r"""This control method is called from ``Env`` on each ``step``.
        """
        if not niu1niu:
            init_state = self._sim.get_agent_state()

            cur_path = []
            forward_action = HabitatSimActions.MOVE_FORWARD

            init_forward = self._sim.get_agent(0).agent_config.action_space[
                forward_action].actuation.amount
            theta = np.arctan2(init_state.rotation.imag[1], 
                init_state.rotation.real) + angle / 2
            rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
            self._sim.set_agent_state(init_state.position, rotation)

            ksteps = int(distance//init_forward)
            for k in range(ksteps):
                if k == ksteps - 1:
                    output = self._sim.step(forward_action)
                else:
                    self._sim.step_without_obs(forward_action)
                cur_path.append(self.get_agent_info())

            output['cur_path'] = cur_path
            return output

        elif niu1niu:
            cur_path = []
            forward_action = HabitatSimActions.MOVE_FORWARD
            left_action = HabitatSimActions.TURN_LEFT
            right_action = HabitatSimActions.TURN_RIGHT
            foward_unit = self._sim.get_agent(0).agent_config.action_space[forward_action].actuation.amount

            angle = math.degrees(angle)
            self.turn(angle)

            forward_step = int(distance // foward_unit)
            for k in range(forward_step):
                if k == forward_step - 1:
                    output = self._sim.step(forward_action)
                    cur_path.append(self.get_agent_info())
                    break
                else:
                    self._sim.step_without_obs(forward_action)
                    cur_path.append(self.get_agent_info())
                    if self._sim.previous_step_collided:
                        output = self._sim.step(forward_action)
                        break
            
            # left forward step
            forward_step = forward_step - len(cur_path)
            if forward_step > 0:
                # assert self._sim.previous_step_collided == True
                init_try_angle = random.choice([90, 270]) # left or right randomly
                self.turn(init_try_angle)

                if init_try_angle == 90: # from left to right
                    turn_seqs = [
                        {'head_turns': [],               'tail_turns': [right_action]*3 }, # 90
                        {'head_turns': [right_action],   'tail_turns': [right_action]*2 }, # 60
                        {'head_turns': [right_action],   'tail_turns': [right_action] }, # 30
                        {'head_turns': [right_action]*2, 'tail_turns': [left_action] }, # -30
                        {'head_turns': [right_action],   'tail_turns': [left_action]*2 }, # -60
                        {'head_turns': [right_action],   'tail_turns': [left_action]*3 }, # -90
                    ]
                elif init_try_angle == 270: # from right to left
                    turn_seqs = [
                        {'head_turns': [],              'tail_turns': [left_action]*3 }, # -90
                        {'head_turns': [left_action],   'tail_turns': [left_action]*2 }, # -60
                        {'head_turns': [left_action],   'tail_turns': [left_action] }, # -30
                        {'head_turns': [left_action]*2, 'tail_turns': [right_action] }, # 30
                        {'head_turns': [left_action],   'tail_turns': [right_action]*2 }, # 60
                        {'head_turns': [left_action],   'tail_turns': [right_action]*3 }, # 90
                    ]
                # try each direction, if pos change, do tail_turns, then do left forward actions
                for turn_seq in turn_seqs:
                    for turn in turn_seq['head_turns']:
                        self._sim.step_without_obs(turn)
                    prev_position = self._sim.get_agent_state().position
                    self._sim.step_without_obs(forward_action)
                    post_posiiton = self._sim.get_agent_state().position
                    # pos change
                    if list(prev_position) != list(post_posiiton):
                        cur_path.append(self.get_agent_info())
                        # do tail_turns
                        for turn in turn_seq['tail_turns']:
                            self._sim.step_without_obs(turn)
                        # do left forward actions
                        for k in range(forward_step):
                            if k == forward_step - 1:
                                output = self._sim.step(forward_action)
                                cur_path.append(self.get_agent_info())
                            else:
                                self._sim.step_without_obs(forward_action)
                                cur_path.append(self.get_agent_info())
                                if self._sim.previous_step_collided:
                                    output = self._sim.step(forward_action)
                                    break
                        break
            
            output['cur_path'] = cur_path
            return output