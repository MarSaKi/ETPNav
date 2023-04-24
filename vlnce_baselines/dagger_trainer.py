import gc
import os
import random
import warnings
from collections import defaultdict

import lmdb
import msgpack_numpy
import numpy as np
import torch
import torch.nn.functional as F
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
from vlnce_baselines.common.env_utils import construct_envs, is_slurm_batch_job
from vlnce_baselines.common.utils import extract_instruction_tokens, get_camera_orientations12

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401

import torch.distributed as distr
import gzip
import json
from copy import deepcopy

class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self


def collate_fn(batch):
    
    instruction_enc = [traj_obs[0]['instruction_enc'] for traj_obs in batch]
    instruction_enc = torch.stack(instruction_enc, dim=0)
    lang_masks = (instruction_enc != 0)

    batch_size = len(batch)
    traj_lens = [len(traj_obs) for traj_obs in batch]
    max_traj_len = max(traj_lens)
    max_cand_len = 6 # defined by waypoint predictor
    rgb_features_dim = batch[0][0]['rgb_features'].shape[-1]
    depth_features_dim = batch[0][0]['depth_features'].shape[-1]
    cand_direction_dim = batch[0][0]['cand_direction'].shape[-1]

    rgb_features = torch.zeros([batch_size, max_traj_len, max_cand_len, rgb_features_dim])
    depth_features = torch.zeros([batch_size, max_traj_len, max_cand_len, depth_features_dim])
    cand_direction = torch.zeros([batch_size, max_traj_len, max_cand_len, cand_direction_dim])
    cand_masks = torch.ones([batch_size, max_traj_len, max_cand_len]).bool()
    action = torch.ones([batch_size, max_traj_len, 1]).long() * -100 # -100 means ended, ignored
    prev_action = torch.zeros([batch_size, max_traj_len, 1])

    for t in range(max_traj_len):
        for i in range(batch_size):
            if t < traj_lens[i]: # traj_i not ended
                cur_cand_len = batch[i][t]['cand_direction'].shape[0]
                rgb_features[i, t, 0:cur_cand_len, :] = batch[i][t]['rgb_features']
                depth_features[i, t, 0:cur_cand_len, :] = batch[i][t]['depth_features']
                cand_direction[i, t, 0:cur_cand_len, :] = batch[i][t]['cand_direction']
                cand_masks[i, t, 0:cur_cand_len] = False
                action[i, t, :] = batch[i][t]['action']
                prev_action[i, t, :] = batch[i][t]['prev_action']

    return (
            instruction_enc,
            lang_masks,
            rgb_features,
            depth_features,
            cand_direction,
            cand_masks,
            action,
            prev_action,
            traj_lens
        )


def _block_shuffle(lst, block_size):
    blocks = [lst[i : i + block_size] for i in range(0, len(lst), block_size)]
    random.shuffle(blocks)

    return [ele for block in blocks for ele in block]


class VLNBERTTrajectoryDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        lmdb_features_dir,
        lmdb_map_size=1e9,
        batch_size=1,
        # for distributed:
        is_distributed=False,
        rank = 0,
        world_size = 1,
    ):
        super().__init__()
        self.lmdb_features_dir = lmdb_features_dir
        self.lmdb_map_size = lmdb_map_size
        self.preload_size = batch_size * 100
        self._preload = []
        self.batch_size = batch_size

        with lmdb.open(self.lmdb_features_dir, map_size=int(self.lmdb_map_size), readonly=True, lock=False) as lmdb_env:
            self.length = lmdb_env.stat()["entries"]
        self.start = 0
        self.end = self.length
        if is_distributed:
            per_rank = int(np.ceil(self.length / world_size))
            self.start = per_rank * rank
            if rank == world_size-1:
                self.end = min(self.start + per_rank, self.length) # the last rank maybe out of index
            else:
                self.end = self.start + per_rank
            self.length = per_rank

    def _load_next(self):
        if len(self._preload) == 0:
            if len(self.load_ordering) == 0:
                raise StopIteration

            new_preload = []
            lengths = []
            with lmdb.open(self.lmdb_features_dir, map_size=int(self.lmdb_map_size), readonly=True, lock=False) as lmdb_env, \
                lmdb_env.begin(buffers=True) as txn:
                for _ in range(self.preload_size):
                    if len(self.load_ordering) == 0:
                        break

                    new_preload.append(msgpack_numpy.unpackb(txn.get(str(self.load_ordering.pop()).encode()), raw=False))
                    lengths.append(len(new_preload[-1][0]))

            sort_priority = list(range(len(lengths)))
            random.shuffle(sort_priority)

            sorted_ordering = list(range(len(lengths)))
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            for idx in _block_shuffle(sorted_ordering, self.batch_size):
                self._preload.append(new_preload[idx])

        return self._preload.pop()

    def __next__(self):
        traj_obs = self._load_next()
        for stepk, obs in enumerate(traj_obs):
            for k, v in obs.items():
                if k in ['action', 'instruction_enc']:
                    traj_obs[stepk][k] = torch.from_numpy(np.copy(v).astype(np.int64))
                else:
                    traj_obs[stepk][k] = torch.from_numpy(np.copy(v).astype(np.float32))

        return traj_obs

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start = self.start
            end = self.end
        else:
            per_worker = int(np.ceil(self.length / worker_info.num_workers))
            start = self.start + per_worker * worker_info.id
            end = min(start + per_worker, self.end)

        # Reverse so we can use .pop()
        self.load_ordering = list(
            reversed(
                _block_shuffle(list(range(start, end)), self.preload_size)
            )
        )
        return self


@baseline_registry.register_trainer(name="dagger")
class DaggerTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        self.lmdb_features_dir = config.IL.DAGGER.lmdb_features_dir.format(split=config.TASK_CONFIG.DATASET.SPLIT)
        super().__init__(config)
        self.max_len = int(config.IL.max_traj_len)

    def _make_dirs(self) -> None:
        os.system(f"mkdir -p {self.lmdb_features_dir}")
        if self.config.local_rank < 1:
            self._make_ckpt_dir()
            if self.config.EVAL.SAVE_RESULTS:
                self._make_results_dir()
            # os.makedirs(self.lmdb_features_dir, exist_ok=True)

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

    def _teacher_action(self, batch_angles, batch_distances, candidate_lengths):
        cand_dists_to_goal = [[] for _ in range(len(batch_angles))]
        oracle_cand_idx = []
        for j in range(len(batch_angles)):
            for k in range(len(batch_angles[j])):
                angle_k = batch_angles[j][k]
                forward_k = batch_distances[j][k]
                dist_k = self.envs.call_at(j, "cand_dist_to_goal", {"angle": angle_k, "forward": forward_k})
                cand_dists_to_goal[j].append(dist_k)
            curr_dist_to_goal = self.envs.call_at(j, "current_dist_to_goal")
            # if within target range (which def as 3.0)
            if curr_dist_to_goal < 1.5:
                oracle_cand_idx.append(candidate_lengths[j] - 1)
            else:
                oracle_cand_idx.append(np.argmin(cand_dists_to_goal[j]))
        return oracle_cand_idx

    @torch.no_grad()
    def _collect_batch(self, dagger_ratio):
        self.envs.resume_all()
        observations = self.envs.reset()
        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        not_done_masks = torch.zeros(self.envs.num_envs, 1, dtype=torch.bool, device=self.device)
        not_done_index = list(range(self.envs.num_envs))
        id2episodes = defaultdict(list)

        # encode instructions
        lang_idx_tokens = batch['instruction']
        all_lang_masks = (lang_idx_tokens != 0)
        h_t, all_language_features = self.policy.net(
            mode='language',
            lang_idx_tokens=lang_idx_tokens,
            lang_masks=all_lang_masks,
        )
        _lang_idx_tokens = lang_idx_tokens.cpu().numpy().astype(np.float16) # for collection

        # build hook
        def hook_builder(tgt_tensor):
            def hook(m, i, o):
                tgt_tensor.set_(o.cpu())
            return hook
        rgb_features = torch.zeros((1,), device="cpu")
        depth_features = torch.zeros((1,), device="cpu")
        if self.world_size > 1:
            rgb_hook = self.policy.net.module.space_pool_rgb.register_forward_hook(hook_builder(rgb_features))
            depth_hook = self.policy.net.module.space_pool_depth.register_forward_hook(hook_builder(depth_features))
        else:
            rgb_hook = self.policy.net.space_pool_rgb.register_forward_hook(hook_builder(rgb_features))
            depth_hook = self.policy.net.space_pool_depth.register_forward_hook(hook_builder(depth_features))

        for stepk in range(self.max_len):
            language_features = all_language_features[not_done_index]
            lang_masks = all_lang_masks[not_done_index]
            language_features = torch.cat((h_t.unsqueeze(1), language_features[:,1:,:]), dim=1)

            # agent's current position and heading
            positions = []; headings = []
            for ob_i in range(self.envs.num_envs):
                agent_state_i = self.envs.call_at(ob_i, "get_agent_info", {})
                positions.append(agent_state_i['position'])
                headings.append(agent_state_i['heading'])
            
            # cand waypoint prediction
            cand_rgb, cand_depth, cand_direction, cand_mask, candidate_lengths, \
            batch_angles, batch_distances = self.policy.net(
                mode = "waypoint",
                waypoint_predictor = self.waypoint_predictor,
                observations = batch,
                in_train = self.config.IL.waypoint_aug,
            )

            # navigation logit
            logits, h_t = self.policy.net(
                mode = 'navigation',
                observations=batch,
                lang_masks=lang_masks,
                lang_feats=language_features,
                headings=headings,
                cand_rgb = cand_rgb, 
                cand_depth = cand_depth,
                cand_direction = cand_direction,
                cand_mask = cand_mask,                    
                masks = not_done_masks,
            )
            logits = logits.masked_fill_(cand_mask, -float('inf'))

            # sample action
            oracle_cand_idx = self._teacher_action(batch_angles, batch_distances, candidate_lengths)
            oracle_actions = torch.tensor(oracle_cand_idx, device=self.device).unsqueeze(1)
            actions = logits.argmax(dim=-1, keepdim=True)
            actions = torch.where(torch.rand_like(actions, dtype=torch.float)<=dagger_ratio, oracle_actions, actions)

            # update rgb & depth features
            _rgb_features = rgb_features.numpy().astype(np.float16)
            _depth_features = depth_features.numpy().astype(np.float16)
            _cand_direction = cand_direction.cpu().numpy().astype(np.float16)
            for j in range(self.envs.num_envs):
                rgb_keys = [k for k in observations[j].keys() if 'rgb' in k]
                depth_keys = [k for k in observations[j].keys() if 'depth' in k]
                for k in rgb_keys:
                    del observations[j][k]
                for k in depth_keys:
                    del observations[j][k]
                del observations[j]['instruction']
                del observations[j]['shortest_path_sensor']
                del observations[j]['progress']
                t_candidate_len = candidate_lengths[j]
                observations[j]['rgb_features'] = _rgb_features[j, 0:t_candidate_len, :]
                observations[j]['depth_features'] = _depth_features[j, 0:t_candidate_len, :]
                observations[j]['cand_direction'] = _cand_direction[j, 0:t_candidate_len, :]
                observations[j]['instruction_enc'] = _lang_idx_tokens[j]
                observations[j]['action'] = np.array([oracle_cand_idx[j]]).astype(np.float16) # teacher action
                observations[j]['prev_action'] = np.array([headings[j]]).astype(np.float16)
            for j, ep in enumerate(self.envs.current_episodes()):
                id2episodes[ep.episode_id].append(observations[j])

            # make equiv action
            env_actions = []
            for j in range(self.envs.num_envs):
                if actions[j].item()==candidate_lengths[j]-1 or stepk==self.max_len-1:
                    env_actions.append({'action':{'action': 0, 'action_args':{}}})
                else:
                    t_angle = batch_angles[j][actions[j].item()]
                    t_distance = batch_distances[j][actions[j].item()]
                    env_actions.append({'action':{'action': 4, 'action_args':{'angle': t_angle, 'distance': t_distance}}})
            outputs = self.envs.step(env_actions)
            observations, _, dones, _ = [list(x) for x in zip(*outputs)]

            # pause env
            if sum(dones) > 0:
                shift_index = 0
                for i in range(self.envs.num_envs):
                    if dones[i]:
                        i = i - shift_index
                        not_done_index.pop(i)
                        self.envs.pause_at(i)
                        if self.envs.num_envs == 0:
                            break
                        observations.pop(i)
                        shift_index += 1
            if self.envs.num_envs == 0:
                break

            not_done_masks = torch.zeros(self.envs.num_envs, 1, dtype=torch.bool, device=self.device)
            h_t = h_t[np.array(dones)==False]
            observations = extract_instruction_tokens(observations,self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if rgb_hook is not None:
            rgb_hook.remove()
        if depth_hook is not None:
            depth_hook.remove()
        return list(id2episodes.values())

    def _update_dataset(self, data_it):
        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

        self.policy.eval()
        self.waypoint_predictor.eval()

        collected_eps = 0
        p = self.config.IL.DAGGER.p
        # in Python 0.0 ** 0.0 == 1.0, but we want 0.0
        beta = 0.0 if p == 0.0 else p ** (data_it//2)
        lmdb_commit_frequency = 5 # 1 commit == lmdb_commit_frequency * NUM_ENVIRONMENTS * world_size
        batch_iter = 0

        if self.local_rank < 1:
            pbar = tqdm.tqdm(total=self.config.IL.DAGGER.update_size, dynamic_ncols=True)
            lmdb_env = lmdb.open(self.lmdb_features_dir, map_size=int(self.config.IL.DAGGER.lmdb_map_size))
            start_id = lmdb_env.stat()["entries"]
            txn = lmdb_env.begin(write=True)

        while collected_eps < self.config.IL.DAGGER.update_size:
            episodes = self._collect_batch(beta)

            # gather episodes
            if self.world_size > 1:
                gather_episodes = [None for _ in range(self.world_size)]
                distr.all_gather_object(gather_episodes, episodes)
                merge_episodes = []
                for x in gather_episodes:
                    merge_episodes += x
                episodes = merge_episodes

            batch_iter += 1
            collected_eps += len(episodes)

            if self.local_rank < 1:
                for traj_obs in episodes:
                    txn.put(str(start_id).encode(), msgpack_numpy.packb(traj_obs, use_bin_type=True))
                    start_id += 1 # only rank0 update
                pbar.update(len(episodes))
                if batch_iter % lmdb_commit_frequency == 0:
                    txn.commit()
                    txn = lmdb_env.begin(write=True)

        if self.local_rank < 1:
            lmdb_env.close()

    def _update_agent(self, instruction_enc, lang_masks, 
                    rgb_features, depth_features, cand_direction, cand_masks,
                    action, prev_action, traj_lens):
        self.policy.train()
        ml_loss = 0.

        h_t, lang_features = self.policy.net(
            mode='language',
            lang_idx_tokens=instruction_enc,
            lang_masks=lang_masks,
        )

        max_traj_len = max(traj_lens)
        for t in range(max_traj_len):
            lang_features = torch.cat((h_t.unsqueeze(1), lang_features[:,1:,:]), dim=1)
            logits, h_t = self.policy.net(
                    mode = 'navigation',
                    lang_masks=lang_masks,
                    lang_feats=lang_features,
                    headings=prev_action[:, t],
                    cand_rgb = rgb_features[:, t, ...], 
                    cand_depth = depth_features[:, t, ...],
                    cand_direction = cand_direction[:, t, ...],
                    cand_mask = cand_masks[:, t, ...],                    
                )

            logits = logits.masked_fill_(cand_masks[:, t, ...], -float('inf'))
            target = action[:, t].squeeze(1)
            t_loss = F.cross_entropy(logits, target, reduction='none', ignore_index=-100)
            ml_loss += torch.sum(t_loss)
        
        total_actions = sum(traj_lens)
        ml_loss = ml_loss / total_actions

        self.optimizer.zero_grad()
        ml_loss.backward()
        if self.world_size > 1:
            torch.nn.utils.clip_grad_norm_(self.policy.net.module.vln_bert.parameters(), 40.)
        else:
            torch.nn.utils.clip_grad_norm_(self.policy.net.vln_bert.parameters(), 40.)
        self.optimizer.step()

        return ml_loss.item()

    def _set_config(self):
        self.split = self.config.TASK_CONFIG.DATASET.SPLIT
        self.config.defrost()
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.split
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.split
        self.config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = self.config.IL.max_traj_len
        if self.config.IL.DAGGER.expert_policy_sensor not in self.config.TASK_CONFIG.TASK.SENSORS:
            self.config.TASK_CONFIG.TASK.SENSORS.append(self.config.IL.DAGGER.expert_policy_sensor)
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.SIMULATOR_GPU_IDS = self.config.SIMULATOR_GPU_IDS[self.config.local_rank]
        self.config.use_pbar = not is_slurm_batch_job()
        ''' if choosing image '''
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
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

    def _init_envs(self):
        # for DDP to load different data
        self.config.defrost()
        self.config.TASK_CONFIG.SEED = self.config.TASK_CONFIG.SEED + self.local_rank
        self.config.freeze()

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False
        )
        env_num = self.envs.num_envs
        dataset_len = sum(self.envs.number_of_episodes)
        logger.info(f'LOCAL RANK: {self.local_rank}, ENV NUM: {env_num}, DATASET LEN: {dataset_len}')
        observation_space = self.envs.observation_spaces[0]
        action_space = self.envs.action_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        return observation_space, action_space

    def train(self) -> None:
        """Main method for training DAgger."""
        self._set_config() #TODO set proxy SAP dataset
        if self.local_rank < 1:
            if self.config.IL.DAGGER.preload_lmdb_features:
                try:
                    lmdb.open(self.lmdb_features_dir, readonly=True)
                except lmdb.Error as err:
                    logger.error("Cannot open database for teacher forcing preload.")
                    raise err
            else:
                with lmdb.open(self.lmdb_features_dir, map_size=int(self.config.IL.DAGGER.lmdb_map_size)) as lmdb_env, \
                    lmdb_env.begin(write=True) as txn:
                    txn.drop(lmdb_env.open_db())

        observation_space, action_space = self._init_envs()
        print('\nInitializing policy network ...')
        self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )

        print('\nTraining starts ...')
        with TensorboardWriter(self.config.TENSORBOARD_DIR if self.local_rank < 1 else '', flush_secs=self.flush_secs, purge_step=0) as writer:
            for dagger_it in range(self.config.IL.DAGGER.iterations):
                step_id = 0
                if not self.config.IL.DAGGER.preload_lmdb_features:
                    self._update_dataset(dagger_it + (1 if self.config.IL.load_from_ckpt else 0))

                if torch.cuda.is_available():
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
                gc.collect()

                if self.world_size > 1:
                    dataset = VLNBERTTrajectoryDataset(self.lmdb_features_dir, 
                                                       batch_size=self.config.IL.batch_size, 
                                                       is_distributed=True, 
                                                       rank=self.local_rank,
                                                       world_size=self.world_size)
                else:
                    dataset = VLNBERTTrajectoryDataset(self.lmdb_features_dir, batch_size=self.config.IL.batch_size)
                diter = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.config.IL.batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                    pin_memory=False,
                    drop_last=True,  # drop last batch if smaller
                    num_workers=3,
                )

                for epoch in range(self.config.IL.epochs):
                    if self.config.use_pbar and self.local_rank < 1:
                        pbar = tqdm.tqdm(total = dataset.length // dataset.batch_size, dynamic_ncols=True)
                    else:
                        pbar = range(dataset.length // dataset.batch_size)
                    for batch in diter:
                        (
                            instruction_enc,
                            lang_masks,
                            rgb_features,
                            depth_features,
                            cand_direction,
                            cand_masks,
                            action,
                            prev_action,
                            traj_lens,
                        ) = batch
                        
                        loss = self._update_agent(
                            instruction_enc.to(device=self.device, non_blocking=True),
                            lang_masks.to(device=self.device, non_blocking=True),
                            rgb_features.to(device=self.device, non_blocking=True),
                            depth_features.to(device=self.device, non_blocking=True),
                            cand_direction.to(device=self.device, non_blocking=True),
                            cand_masks.to(device=self.device, non_blocking=True),
                            action.to(device=self.device, non_blocking=True),
                            prev_action.to(device=self.device, non_blocking=True),
                            traj_lens,
                        )
                        
                        if self.local_rank < 1:
                            logger.info(f'dagger iter: {dagger_it}, epoch: {epoch+1} / {self.config.IL.epochs}, loss: {loss:.4f}')
                            writer.add_scalar(f'loss/dagger_it{dagger_it}', loss, step_id)
                            pbar.update()

                        step_id += 1

                    if self.local_rank < 1:    
                        self.save_checkpoint(dagger_it*self.config.IL.epochs+epoch, step_id)
                        
        print("**************** END ****************")
        import pdb;pdb.set_trace()