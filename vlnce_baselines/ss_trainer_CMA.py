import gc
import os
import random
import warnings
from collections import defaultdict

import lmdb
import msgpack_numpy
import numpy as np
import math
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import tqdm
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs, construct_envs_for_rl, is_slurm_batch_job
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.utils import reduce_loss

from .utils import get_camera_orientations12
from .utils import (
    length2mask, dir_angle_feature,
)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401

import torch.distributed as distr
import gzip
import json
from copy import deepcopy

@baseline_registry.register_trainer(name="schedulesampler-CMA")
class SSTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.max_len = int(config.IL.max_traj_len) #  * 0.97 transfered gt path got 0.96 spl

    def _make_dirs(self) -> None:
        if self.config.local_rank == 0:
            self._make_ckpt_dir()
            # os.makedirs(self.lmdb_features_dir, exist_ok=True)
            if self.config.EVAL.SAVE_RESULTS:
                self._make_results_dir()

    def save_checkpoint(self, epoch: int, step_id: int) -> None:
        torch.save(
            obj={
                "state_dict": self.policy.state_dict(),
                # "waypoint_predictor_state_dict": self.waypoint_predictor.state_dict(),
                "config": self.config,
                "optim_state": self.optimizer.state_dict(),
                # "way_rl_optim_state": self.way_rl_optimizer.state_dict(),
                "epoch": epoch,
                "step_id": step_id,
            },
            f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.{epoch}.pth"),
        )

    def allocate_allowed_episode_by_scene(self):
        ''' discrete waypoints coordinates directly projected from MP3D '''
        if 'R2R' in self.config.TASK_CONFIG.DATASET.DATA_PATH:
            with gzip.open(
                self.config.TASK_CONFIG.DATASET.DATA_PATH.format(
                    split=self.split)
            ) as f:
                data = json.load(f) # dict_keys(['episodes', 'instruction_vocab'])

            ''' continuous waypoints coordinates by shortest paths in Habitat '''
            with gzip.open(
                self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                    split=self.split)
            ) as f:
                gt_data = json.load(f)
            
        elif 'RxR' in self.config.TASK_CONFIG.DATASET.DATA_PATH:
            gt_data = {}
            data = {}
            for role in self.config.TASK_CONFIG.DATASET.ROLES:
                with gzip.open(
                    self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                        split=self.split, role=role
                    ),
                    "rt",
                ) as f:
                    gt_data.update(json.load(f))
                with gzip.open(
                    self.config.TASK_CONFIG.DATASET.DATA_PATH.format(
                        split=self.split, role=role
                    ),
                    "rt",
                ) as f:
                    data.update(json.load(f))

        # data['episodes'][K] corresponds to
        # gt_data[str(data['episodes'][K]['episode_id'])]
        # print('checked dataset')
        # import pdb; pdb.set_trace()

        data = data['episodes']
        long_episode_ids = []
        # if 'RxR' in  self.config.TASK_CONFIG.DATASET.DATA_PATH:
        #     long_episode_ids = [int(k) for k,v in gt_data.items() if len(v['actions']) > self.config.IL.max_traj_len]
        average_length = (len(data) - len(long_episode_ids)) // self.world_size

        episodes_by_scene = {}
        for ep in data:
            scan = ep['scene_id'].split('/')[1]
            if scan not in episodes_by_scene.keys():
                episodes_by_scene[scan] = []
            if ep['episode_id'] not in long_episode_ids:
                episodes_by_scene[scan].append(ep['episode_id'])
            else:
                continue

        ''' split data in each environments evenly to different GPUs ''' # averaging number set problem
        values_to_scenes = {}
        values = []
        for k,v in episodes_by_scene.items():
            values.append(len(v))
            if len(v) not in values_to_scenes.keys():
                values_to_scenes[len(v)] = []
            values_to_scenes[len(v)].append(k)

        groups = self.world_size
        values.sort(reverse=True)
        last_scene_episodes = episodes_by_scene[values_to_scenes[values[0]].pop()]
        values = values[1:]

        load_balance_groups = [[] for grp in range(groups)]
        scenes_groups = [[] for grp in range(groups)]

        for v in values:
            current_total = [sum(grp) for grp in load_balance_groups]
            min_index = np.argmin(current_total)
            load_balance_groups[min_index].append(v)
            scenes_groups[min_index] += episodes_by_scene[values_to_scenes[v].pop()]

        for grp in scenes_groups:
            add_number = average_length - len(grp)
            grp += last_scene_episodes[:add_number]
            last_scene_episodes = last_scene_episodes[add_number:]

        # episode_ids = [ep['episode_id'] for ep in data if
        #                ep['episode_id'] not in long_episode_ids]
        # scenes_groups[self.local_rank] = episode_ids[
        #                 self.local_rank:self.world_size * average_length:self.world_size]
        self.gt_data = gt_data
        return scenes_groups[self.local_rank]

    def train_ml(self, in_train=True, train_tf=False, train_rl=False):
        self.envs.resume_all()
        observations = self.envs.reset()

        shift_index = 0
        for i, ep in enumerate(self.envs.current_episodes()):
            if ep.episode_id in self.trained_episodes:
                i = i - shift_index
                observations.pop(i)
                self.envs.pause_at(i)
                shift_index += 1
                if self.envs.num_envs == 0:
                    break
            else:
                self.trained_episodes.append(ep.episode_id)

        if self.envs.num_envs == 0:
            return -1

        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        # expert_uuid = self.config.IL.DAGGER.expert_policy_sensor_uuid

        # prev_actions = torch.zeros(
        #     self.envs.num_envs, 1, device=self.device, dtype=torch.long
        # )
        not_done_masks = torch.zeros(
            self.envs.num_envs, 1, dtype=torch.bool, device=self.device
        )
        # prev_oracle_actions = prev_actions.clone()

        ml_loss = 0.
        total_weight = 0.
        losses = []
        # weights = []
        not_done_index = list(range(self.envs.num_envs))

        # encoding instructions
        if 'CMA' in self.config.MODEL.policy_name:
            rnn_states = torch.zeros(
                self.envs.num_envs,
                self.num_recurrent_layers,
                self.config.MODEL.STATE_ENCODER.hidden_size,
                device=self.device,
            )
            instruction_embedding, all_lang_masks = self.policy.net(
                mode = "language",
                observations = batch,
            )
        init_num_envs = self.envs.num_envs

        # # Init the reward shaping
        # last_dist = np.zeros(len(observations), np.float32)
        # last_ndtw = np.zeros(len(observations), np.float32)
        # for i in range(len(observations)):
        #     info = self.envs.call_at(i, "get_metrics", {})
        #     last_dist[i] = info['distance_to_goal']
            # last_ndtw[i] = info['ndtw']
        # init_bs = len(observations)
        # state_not_dones = np.array([True] * init_bs)
        # rewards = []
        # hidden_states = []
        # policy_log_probs = []
        # critic_masks = []
        # entropys = []

        il_loss = 0.0
        for stepk in range(self.max_len):
            language_features = instruction_embedding[not_done_index]
            lang_masks = all_lang_masks[not_done_index]

            # agent's current position and heading
            positions = []; headings = []
            for ob_i in range(len(observations)):
                agent_state_i = self.envs.call_at(ob_i,
                        "get_agent_info", {})
                positions.append(agent_state_i['position'])
                headings.append(agent_state_i['heading'])

            if 'CMA' in self.config.MODEL.policy_name:
                # candidate waypoints prediction
                cand_rgb, cand_depth, \
                cand_direction, cand_mask, candidate_lengths, \
                batch_angles, batch_distances = self.policy.net(
                    mode = "waypoint",
                    waypoint_predictor = self.waypoint_predictor,
                    observations = batch,
                    in_train = in_train,
                )
                # navigation action logits
                logits, rnn_states = self.policy.net(
                    mode = 'navigation',
                    observations = batch,
                    instruction = language_features,
                    text_mask = lang_masks,
                    rnn_states = rnn_states,
                    headings = headings,
                    cand_rgb = cand_rgb, 
                    cand_depth = cand_depth,
                    cand_direction = cand_direction,
                    cand_mask = cand_mask,
                    masks = not_done_masks,
                )
                # step_rnn_states = torch.zeros(init_bs, 512, device=self.device)
                # step_rnn_states[state_not_dones] = rnn_states.squeeze(1)
                # hidden_states.append(step_rnn_states)

            logits = logits.masked_fill_(cand_mask, -float('inf'))
            total_weight += len(candidate_lengths)

            # get resulting distance by execting candidate actions
            # the last value in each list is the current distance
            if train_tf:
                # shortest path oracle
                if 'R2R' in self.config.TASK_CONFIG.DATASET.DATA_PATH:
                    cand_dists_to_goal = [[] for _ in range(len(batch_angles))]
                    oracle_cand_idx = []
                    oracle_stop = []
                    for j in range(len(batch_angles)):
                        for k in range(len(batch_angles[j])):
                            angle_k = batch_angles[j][k]
                            forward_k = batch_distances[j][k]
                            dist_k = self.envs.call_at(j, 
                                "cand_dist_to_goal", {
                                    "angle": angle_k, "forward": forward_k,
                                })
                            cand_dists_to_goal[j].append(dist_k)
                        curr_dist_to_goal = self.envs.call_at(
                            j, "current_dist_to_goal")
                        # if within target range (which def as 3.0)
                        if curr_dist_to_goal < 1.5:
                            oracle_cand_idx.append(candidate_lengths[j] - 1)
                            oracle_stop.append(True)
                        else:
                            oracle_cand_idx.append(np.argmin(cand_dists_to_goal[j]))
                            oracle_stop.append(False)
                # subgoal oracle
                elif 'RxR' in self.config.TASK_CONFIG.DATASET.DATA_PATH:
                    kargs = []
                    current_episodes = self.envs.current_episodes()
                    for karg_i in range(self.envs.num_envs):
                        kargs.append({
                            'ref_path':self.gt_data[str(current_episodes[karg_i].episode_id)]['locations'],
                            'angles':batch_angles[karg_i],
                            'distances':batch_distances[karg_i],
                            'candidate_length':candidate_lengths[karg_i]
                            })
                    oracle_cand_idx = self.envs.call(["get_cand_idx"]*self.envs.num_envs,kargs)
                else:
                    NotImplementedError

            if train_rl:
                raise NotImplementedError
                # probs = F.softmax(logits, 1)  # sampling an action from model
                # c = torch.distributions.Categorical(probs)
                # actions = c.sample().detach()
                # rl_entropy = torch.zeros(init_bs, device=self.device)
                # rl_entropy[state_not_dones] = c.entropy()
                # entropys.append(rl_entropy)
                # rl_policy_log_probs = torch.zeros(init_bs, device=self.device)
                # rl_policy_log_probs[state_not_dones] = c.log_prob(actions)
                # policy_log_probs.append(rl_policy_log_probs)
            elif train_tf:
                oracle_actions = torch.tensor(oracle_cand_idx, device=self.device).unsqueeze(1)
                actions = logits.argmax(dim=-1, keepdim=True)
                actions = torch.where(
                        torch.rand_like(actions, dtype=torch.float) <= self.ratio,
                        oracle_actions, actions)
                current_loss = F.cross_entropy(logits, oracle_actions.squeeze(1), reduction="none")
                ml_loss += torch.sum(current_loss)
            else:
                actions = logits.argmax(dim=-1, keepdim=True)

            # action_angles = []
            # action_distances = []
            env_actions = []
            # rl_actions = np.array([-100] * init_bs)
            for j in range(logits.size(0)):
                if train_rl and (actions[j].item() == candidate_lengths[j]-1 or stepk == self.max_len-1):
                    # if RL, force stop at the max step
                    env_actions.append({'action':
                        {'action': 0, 'action_args':{}}})
                elif actions[j].item() == candidate_lengths[j]-1:
                    env_actions.append({'action':
                        {'action': 0, 'action_args':{}}})
                else:
                    env_actions.append({'action':
                        {'action': 4,  # HIGHTOLOW
                        'action_args':{
                            'angle': batch_angles[j][actions[j].item()], 
                            'distance': batch_distances[j][actions[j].item()],
                        }}})

            # self.envs.step(env_actions)
            outputs = self.envs.step(env_actions)
            observations, _, dones, infos = [list(x) for x in
                                             zip(*outputs)]
            
            # if train_rl:
            #     rl_actions[state_not_dones] = np.array([sk['action']['action'] for sk in env_actions])
            #     # Calculate the mask and reward
            #     current_dist = np.zeros(init_bs, np.float32)
            #     # ndtw_score = np.zeros(init_bs, np.float32)
            #     reward = np.zeros(init_bs, np.float32)
            #     ct_mask = np.ones(init_bs, np.float32)

            #     sbi = 0
            #     for si in range(init_bs):
            #         if state_not_dones[si]:
            #             info = self.envs.call_at(sbi, "get_metrics", {})
            #             current_dist[si] = info['distance_to_goal']
            #             # ndtw_score[si] = info['ndtw']
            #             sbi += 1

            #         if not state_not_dones[si]:
            #             reward[si] = 0.0
            #             ct_mask[si] = 0.0
            #         else:
            #             action_idx = rl_actions[si]
            #             # Target reward
            #             if action_idx == 0:                              # If the action now is end
            #                 if current_dist[si] < 3.0:                    # Correct
            #                     reward[si] = 2.0 # + ndtw_score[si] * 2.0
            #                 else:                                         # Incorrect
            #                     reward[si] = -2.0
            #             elif action_idx != -100:                                             # The action is not end
            #                 # Path fidelity rewards (distance & nDTW)
            #                 reward[si] = - (current_dist[si] - last_dist[si])
            #                 # ndtw_reward = ndtw_score[si] - last_ndtw[si]
            #                 if reward[si] > 0.0:                           # Quantification
            #                     reward[si] = 1.0  # + ndtw_reward
            #                 else:
            #                     reward[si] = -1.0 # + ndtw_reward
            #                 # # Miss the target penalty
            #                 # if (last_dist[i] <= 1.0) and (current_dist[i]-last_dist[i] > 0.0):
            #                 #     reward[i] -= (1.0 - last_dist[i]) * 2.0
            #     rewards.append(reward)
            #     critic_masks.append(ct_mask)
            #     last_dist[:] = current_dist
            #     # last_ndtw[:] = ndtw_score

            # state_not_dones[state_not_dones] = np.array(dones) == False

            if sum(dones) > 0:
                if 'CMA' in self.config.MODEL.policy_name:
                    rnn_states = rnn_states[np.array(dones)==False]

                shift_index = 0
                for i in range(self.envs.num_envs):
                    if dones[i]:
                        # print(k, self.local_rank)
                        i = i - shift_index
                        not_done_index.pop(i)
                        self.envs.pause_at(i)
                        if self.envs.num_envs == 0:
                            break
                        observations.pop(i)
                        infos.pop(i)
                        shift_index += 1

            if self.envs.num_envs == 0: # all envs are paused
                break
            not_done_masks = torch.ones(
                self.envs.num_envs, 1, dtype=torch.bool, device=self.device
            )

            observations = extract_instruction_tokens(
                observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            )

            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        # A2C
        if train_rl:
            None
            # rl_loss = 0.
            # length = len(rewards)
            # discount_reward = np.zeros(init_bs, np.float32)
            # rl_total = 0
            # for t in range(length-1, -1, -1):
            #     discount_reward = discount_reward * 0.90 + rewards[t]  # If it ended, the reward will be 0
            #     mask_ = Variable(torch.from_numpy(critic_masks[t]), requires_grad=False).to(self.device)
            #     clip_reward = discount_reward.copy()
            #     r_ = Variable(torch.from_numpy(clip_reward), requires_grad=False).to(self.device)
            #     v_ = self.policy.net(
            #         mode = 'critic',
            #         post_states = hidden_states[t])
            #     a_ = (r_ - v_).detach()
            #     rl_loss += (-policy_log_probs[t] * a_ * mask_).sum()
            #     rl_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5  # 1/2 L2 loss
            #     rl_loss += (- 0.01 * entropys[t] * mask_).sum()
            #     rl_total = rl_total + np.sum(critic_masks[t])
            # rl_loss = rl_loss / rl_total
            # il_loss += rl_loss

        elif train_tf:
            il_loss = ml_loss / total_weight # 0.20 factor

        return il_loss  #, way_rl_loss

    def train(self) -> None:
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        
        self.config.defrost()
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = split
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = split
        self.config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = self.config.IL.max_traj_len
        if (
            self.config.IL.DAGGER.expert_policy_sensor
            not in self.config.TASK_CONFIG.TASK.SENSORS
        ):
            self.config.TASK_CONFIG.TASK.SENSORS.append(
                self.config.IL.DAGGER.expert_policy_sensor
            )
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        self.config.NUM_ENVIRONMENTS = self.config.IL.batch_size // len(
            self.config.SIMULATOR_GPU_IDS)
        self.config.use_pbar = not is_slurm_batch_job()
        ''' *** if choosing image '''
        if self.config.MODEL.policy_name == 'PolicyViewSelectionCMA':
            resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
            crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
            config = self.config.TASK_CONFIG
            camera_orientations = get_camera_orientations12()
            # sensor_uuids = []
            for sensor_type in ["RGB", "DEPTH"]:
                resizer_size = dict(resize_config)[sensor_type.lower()]
                cropper_size = dict(crop_config)[sensor_type.lower()]
                sensor = getattr(config.SIMULATOR, f"{sensor_type}_SENSOR")
                for action, orient in camera_orientations.items():
                    camera_template = f"{sensor_type}_{action}"
                    camera_config = deepcopy(sensor)
                    camera_config.ORIENTATION = camera_orientations[action]
                    camera_config.UUID = camera_template.lower()
                    # sensor_uuids.append(camera_config.UUID)
                    setattr(config.SIMULATOR, camera_template, camera_config)
                    config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                    resize_config.append((camera_template.lower(), resizer_size))
                    crop_config.append((camera_template.lower(), cropper_size))
            self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
            self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
            self.config.TASK_CONFIG = config
            self.config.SENSORS = config.SIMULATOR.AGENT_0.SENSORS
        self.config.freeze()
        
        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        self.batch_size = self.config.IL.batch_size
        torch.cuda.set_device(self.device)
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
            torch.cuda.set_device(self.device)
            
        self.split = split
        episode_ids = self.allocate_allowed_episode_by_scene()

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=episode_ids,
            auto_reset_done=False
        )
        num_epoches_per_ratio = int(np.ceil(self.config.IL.epochs/self.config.IL.decay_time))
        print('\nFinished constructing environments')

        dataset_length = sum(self.envs.number_of_episodes)
        print('\nlocal rank:', self.local_rank, '|', 'dataset length:', dataset_length)

        observation_space = self.envs.observation_spaces[0]
        action_space = self.envs.action_spaces[0]

        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )
        # self.inflection_weight = torch.tensor([1.0,
        #             self.config.IL.inflection_weight_coef], device=self.device)

        print('\nInitializing policy network ...')
        self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )

        print('\nTraining starts ...')
        with TensorboardWriter(
            self.config.TENSORBOARD_DIR if self.local_rank < 1 else '',
            flush_secs=self.flush_secs,
            purge_step=0,
        ) as writer:
            AuxLosses.activate()
            batches_per_epoch = int(np.ceil(dataset_length/self.batch_size))

            for epoch in range(self.start_epoch, self.config.IL.epochs):
                epoch_str = f"{epoch + 1}/{self.config.IL.epochs}"

                t_ = (
                    tqdm.trange(
                        batches_per_epoch, leave=False, dynamic_ncols=True
                    )
                    if self.config.use_pbar & (self.local_rank < 1)
                    else range(batches_per_epoch)
                )
                self.ratio = np.power(self.config.IL.schedule_ratio, epoch//num_epoches_per_ratio + 1)

                self.trained_episodes = []
                # reconstruct env for every epoch to ensure load same data
                if epoch != self.start_epoch:
                    self.envs = None
                    self.envs = construct_envs(
                        self.config, get_env_class(self.config.ENV_NAME),
                        episodes_allowed=episode_ids,
                        auto_reset_done=False
                    )

                for batch_idx in t_:
                    # if batch_idx % 2 == 0:
                    #     loss = self.train_ml(train_rl=False)
                    #     if batch_idx != len(t_)-1:
                    #         continue
                    # else:
                    loss = self.train_ml(  #, way_rl_loss
                        in_train=True, 
                        train_tf=True, train_rl=False)
                    # loss += self.train_ml(train_rl=False)

                    if loss == -1:
                        break
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    losses = [loss]
                    # self.way_rl_optimizer.zero_grad()
                    # way_rl_loss.backward()
                    # self.way_rl_optimizer.step()

                    if self.world_size > 1:
                        for i in range(len(losses)):
                            reduce_loss(losses[i], self.local_rank, self.world_size)
                            losses[i] = losses[i].item()
                    else:
                        for i in range(len(losses)):
                            losses[i] = losses[i].item()
                    loss = losses[0]
                    
                    if self.local_rank < 1:
                        writer.add_scalar("loss", loss, self.step_id)
                        if self.config.use_pbar:
                            t_.set_postfix(
                                {
                                    "epoch": epoch_str,
                                    "loss": round(loss, 4),
                                }
                            )
                    self.step_id += 1  # noqa: SIM113

                if self.local_rank < 1:  # and epoch % 3 == 0:
                    self.save_checkpoint(epoch, self.step_id)

                AuxLosses.deactivate()
