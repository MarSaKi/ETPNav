"""ETPNav + bidirectional world models + corrective memory (IEVC loop).

Trainer name: SS-ETP-WM. The ETPNav policy is loaded from WM.backbone_ckpt
and frozen by default; only the light-weight closed-loop modules in
vlnce_baselines/wm are trained. See docs/WM_DESIGN.md.
"""

import glob
import json
import os
import random
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as distr
import torch.nn.functional as F
import tqdm
from fastdtw import fastdtw
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import apply_obs_transforms_batch
from habitat_baselines.utils.common import batch_obs
from torch.cuda.amp import autocast

from habitat_extensions.measures import NDTW
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.models.graph_utils import GraphMap
from vlnce_baselines.ss_trainer_ETP import RLTrainer
from vlnce_baselines.wm import BiWorldModelNav, ClosedLoopRuntime
from vlnce_baselines.wm.metrics import deviation_report

LOSS_KEYS = ['nav', 'fwd_value', 'inv_goal', 'inv_cycle', 'inv_hind',
             'wm_kl', 'wm_rec', 'wm_pred', 'exec', 'dev', 'corr']


def _strip_module(state_dict):
    return {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}


@baseline_registry.register_trainer(name="SS-ETP-WM")
class WMTrainer(RLTrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.wm_cfg = config.WM
        self.dev_records = []

    # ------------------------------------------------------------ checkpoint
    def save_checkpoint(self, iteration: int):
        obj = {
            "wm_state_dict": self.wm.state_dict(),
            "config": self.config,
            "optim_state": self.optimizer.state_dict(),
            "iteration": iteration,
        }
        if not self.wm_cfg.freeze_backbone:
            obj["state_dict"] = self.policy.state_dict()
        torch.save(obj, os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.iter{iteration}.pth"))

    def _initialize_policy(self, config, load_from_ckpt, observation_space, action_space):
        start_iter = 0
        policy = baseline_registry.get_policy(config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config, observation_space=observation_space, action_space=action_space,
        )
        from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
        self.waypoint_predictor = BinaryDistPredictor_TRM(device=self.device)
        cwp_fn = 'data/wp_pred/check_cwp_bestdist_hfov63' if config.MODEL.task_type == 'rxr' \
            else 'data/wp_pred/check_cwp_bestdist_hfov90'
        self.waypoint_predictor.load_state_dict(
            torch.load(cwp_fn, map_location=torch.device('cpu'))['predictor']['state_dict'])
        for param in self.waypoint_predictor.parameters():
            param.requires_grad_(False)

        # frozen ETPNav backbone (fine-tuned R2R-CE / RxR-CE checkpoint)
        backbone = self.load_checkpoint(self.wm_cfg.backbone_ckpt, map_location="cpu")
        self.policy.load_state_dict(_strip_module(backbone["state_dict"]), strict=False)
        logger.info(f"Loaded ETPNav backbone: {self.wm_cfg.backbone_ckpt}")
        if self.wm_cfg.freeze_backbone:
            for param in self.policy.parameters():
                param.requires_grad_(False)

        self.wm = BiWorldModelNav.from_config(self.wm_cfg)
        self.policy.to(self.device)
        self.waypoint_predictor.to(self.device)
        self.wm.to(self.device)
        self.num_recurrent_layers = self.policy.net.num_recurrent_layers

        param_groups = [{'params': self.wm.parameters(), 'lr': self.wm_cfg.lr}]
        if not self.wm_cfg.freeze_backbone:
            param_groups.append({'params': [p for p in self.policy.parameters() if p.requires_grad],
                                 'lr': config.IL.lr})
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=self.wm_cfg.weight_decay)

        if load_from_ckpt:
            ckpt_path = config.IL.ckpt_to_load
            if config.IL.is_requeue:
                ckpt_list = sorted(filter(os.path.isfile, glob.glob(config.CHECKPOINT_FOLDER + "/*")),
                                   key=os.path.getmtime)
                ckpt_path = ckpt_list[-1] if len(ckpt_list) > 0 else None
            if ckpt_path is not None and os.path.isfile(ckpt_path):
                ckpt = self.load_checkpoint(ckpt_path, map_location="cpu")
                if "wm_state_dict" in ckpt:
                    self.wm.load_state_dict(_strip_module(ckpt["wm_state_dict"]))
                    start_iter = ckpt["iteration"]
                    if config.IL.is_requeue:
                        self.optimizer.load_state_dict(ckpt["optim_state"])
                if "state_dict" in ckpt and not self.wm_cfg.freeze_backbone:
                    self.policy.load_state_dict(_strip_module(ckpt["state_dict"]), strict=False)
                logger.info(f"Loaded WM checkpoint: {ckpt_path}, iteration: {start_iter}")

        self.runtime = ClosedLoopRuntime(self.wm, self.wm_cfg)
        n_wm = sum(p.numel() for p in self.wm.parameters())
        n_all = sum(p.numel() for p in self.policy.parameters()) + n_wm
        logger.info(f"Agent parameters: {n_all/1e6:.2f} M. World-model loop: {n_wm/1e6:.2f} M.")
        return start_iter

    # -------------------------------------------------------------- training
    def _allreduce_grads(self):
        if self.world_size <= 1:
            return
        for p in self.wm.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            distr.all_reduce(p.grad)
            p.grad /= self.world_size

    def _train_interval(self, interval, ml_weight, sample_ratio):
        self.wm.train()
        if self.wm_cfg.freeze_backbone:
            self.policy.eval()
        else:
            self.policy.train()
            self.policy.net.rgb_encoder.eval()
            self.policy.net.depth_encoder.eval()
        self.waypoint_predictor.eval()

        pbar = tqdm.trange(interval, leave=False, dynamic_ncols=True) \
            if self.local_rank < 1 else range(interval)
        self.logs = defaultdict(list)
        for idx in pbar:
            self.optimizer.zero_grad()
            self.loss = 0.
            self.rollout('train', ml_weight, sample_ratio)
            self.loss.backward()
            self._allreduce_grads()
            torch.nn.utils.clip_grad_norm_(self.wm.parameters(), self.wm_cfg.grad_clip)
            self.optimizer.step()
            if self.local_rank < 1:
                pbar.set_postfix({'iter': f'{idx+1}/{interval}'})
        return deepcopy(self.logs)

    # ------------------------------------------------------------ evaluation
    def _eval_checkpoint(self, checkpoint_path, writer, checkpoint_index=0):
        self.dev_records = []
        super()._eval_checkpoint(checkpoint_path, writer, checkpoint_index)
        report = deviation_report(self.dev_records, self.wm_cfg.tau_revise)
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            self.config.RESULTS_DIR,
            f"wm_deviation_ckpt_{checkpoint_index}_{split}_r{self.local_rank}.json")
        with open(fname, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Deviation detection (rank {self.local_rank}): {report}")

    # --------------------------------------------------------------- helpers
    def _backbone(self, **kwargs):
        if self.wm_cfg.freeze_backbone:
            with torch.no_grad(), autocast(enabled=self.wm_cfg.backbone_amp):
                return self.policy.net(**kwargs)
        with autocast(enabled=self.wm_cfg.backbone_amp):
            return self.policy.net(**kwargs)

    def _oracle_dist(self, mode):
        """Geodesic distance to goal, used only as supervision for the
        verifier (train) and for offline deviation metrics (eval)."""
        if mode == 'train' or (mode == 'eval' and self.wm_cfg.log_dev_metrics):
            return [self.envs.call_at(i, "current_dist_to_goal") for i in range(self.envs.num_envs)]
        return None

    def _loss_weight(self, key):
        return float(getattr(self.wm_cfg.LOSS, key))

    # --------------------------------------------------------------- rollout
    def rollout(self, mode, ml_weight=None, sample_ratio=None):
        if mode == 'train':
            feedback = 'sample'
        elif mode == 'eval' or mode == 'infer':
            feedback = 'argmax'
        else:
            raise NotImplementedError

        self.envs.resume_all()
        observations = self.envs.reset()
        instr_max_len = self.config.IL.max_text_len
        instr_pad_id = 1 if self.config.MODEL.task_type == 'rxr' else 0
        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                                                  max_length=instr_max_len, pad_id=instr_pad_id)
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if mode == 'eval':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes())
                            if ep.episode_id in self.stat_eps]
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
        if mode == 'infer':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes())
                            if ep.episode_id in self.path_eps]
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
            curr_eps = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                if self.config.MODEL.task_type == 'rxr':
                    ep_id = curr_eps[i].episode_id
                    k = curr_eps[i].instruction.instruction_id
                    self.inst_ids[ep_id] = int(k)

        all_txt_ids = batch['instruction']
        all_txt_masks = (all_txt_ids != instr_pad_id)
        all_txt_embeds = self._backbone(mode='language', txt_ids=all_txt_ids, txt_masks=all_txt_masks)

        loss = 0.
        loss_terms = defaultdict(float)
        total_actions = 0.
        not_done_index = list(range(self.envs.num_envs))

        have_real_pos = (mode == 'train' or self.config.VIDEO_OPTION)
        ghost_aug = self.config.IL.ghost_aug if mode == 'train' else 0
        self.gmaps = [GraphMap(have_real_pos,
                               self.config.IL.loc_noise,
                               self.config.MODEL.merge_ghost,
                               ghost_aug) for _ in range(self.envs.num_envs)]
        prev_vp = [None] * self.envs.num_envs
        self.runtime.reset(self.envs.num_envs)

        for stepk in range(self.max_len):
            total_actions += self.envs.num_envs
            txt_masks = all_txt_masks[not_done_index]
            txt_embeds = all_txt_embeds[not_done_index]

            wp_outputs = self._backbone(
                mode="waypoint",
                waypoint_predictor=self.waypoint_predictor,
                observations=batch,
                in_train=(mode == 'train' and self.config.IL.waypoint_aug),
            )

            vp_inputs = self._vp_feature_variable(wp_outputs)
            vp_inputs.update({'mode': 'panorama'})
            pano_embeds, pano_masks = self._backbone(**vp_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)

            cur_pos, cur_ori = self.get_pos_ori()
            cur_vp, cand_vp, cand_pos = [], [], []
            for i in range(self.envs.num_envs):
                cur_vp_i, cand_vp_i, cand_pos_i = self.gmaps[i].identify_node(
                    cur_pos[i], cur_ori[i], wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i]
                )
                cur_vp.append(cur_vp_i)
                cand_vp.append(cand_vp_i)
                cand_pos.append(cand_pos_i)

            if mode == 'train' or self.config.VIDEO_OPTION:
                cand_real_pos = []
                for i in range(self.envs.num_envs):
                    cand_real_pos_i = [
                        self.envs.call_at(i, "get_cand_real_pos", {"angle": ang, "forward": dis})
                        for ang, dis in zip(wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i])
                    ]
                    cand_real_pos.append(cand_real_pos_i)
            else:
                cand_real_pos = [None] * self.envs.num_envs

            for i in range(self.envs.num_envs):
                cur_embeds = avg_pano_embeds[i]
                cand_embeds = pano_embeds[i][vp_inputs['nav_types'][i] == 1]
                self.gmaps[i].update_graph(prev_vp[i], stepk + 1,
                                           cur_vp[i], cur_pos[i], cur_embeds,
                                           cand_vp[i], cand_pos[i], cand_embeds,
                                           cand_real_pos[i])

            nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori)
            nav_inputs.update({
                'mode': 'navigation',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
            })
            no_vp_left = nav_inputs.pop('no_vp_left')
            nav_outs = self._backbone(**nav_inputs)
            etp_logits = nav_outs['global_logits'].float()
            valid = torch.isfinite(etp_logits)

            teacher_actions = None
            if mode == 'train' or self.config.VIDEO_OPTION:
                teacher_actions = self._teacher_action_new(nav_inputs['gmap_vp_ids'], no_vp_left)
            oracle_dist = self._oracle_dist(mode)

            # ---- closed loop: verify -> memorise -> correct -> imagine -> fuse
            rt_out = self.runtime.step(
                pano_feat=avg_pano_embeds, txt_embeds=txt_embeds, txt_masks=txt_masks,
                gmap_img_fts=nav_inputs['gmap_img_fts'], gmap_embeds=nav_outs['gmap_embeds'],
                gmap_pos_fts=nav_inputs['gmap_pos_fts'], valid=valid, etp_logits=etp_logits,
                gmaps=self.gmaps, gmap_vp_ids=nav_inputs['gmap_vp_ids'], cur_vp=cur_vp,
                cur_pos=cur_pos, stepk=stepk, max_len=self.max_len,
                teacher_actions=teacher_actions if mode == 'train' else None,
                oracle_dist=oracle_dist, train=(mode == 'train'),
            )
            nav_logits = rt_out['fused_logits']
            nav_probs = F.softmax(nav_logits, 1)
            stop_src = nav_probs if self.wm_cfg.fused_stop_scores else F.softmax(etp_logits, 1)
            for i, gmap in enumerate(self.gmaps):
                gmap.node_stop_scores[cur_vp[i]] = stop_src[i, 0].item() * rt_out['stop_scale'][i]

            if mode == 'train':
                for k, v in rt_out['losses'].items():
                    w = self._loss_weight(k)
                    loss = loss + w * v
                    loss_terms[k] += v.item()

            # determine action
            if feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                a_t = c.sample().detach()
                a_t = torch.where(torch.rand_like(a_t, dtype=torch.float) <= sample_ratio,
                                  teacher_actions, a_t)
                cpu_a_t = a_t.cpu().numpy()
                # off-route perturbation: deviations the memory agent must catch
                for i in range(self.envs.num_envs):
                    if random.random() < self.wm_cfg.perturb_prob:
                        cands = valid[i].nonzero().squeeze(-1).tolist()
                        cands = [k for k in cands if k != 0 and k != int(teacher_actions[i])]
                        if len(cands) > 0:
                            cpu_a_t[i] = random.choice(cands)
            elif feedback == 'argmax':
                cpu_a_t = nav_logits.argmax(dim=-1).cpu().numpy()
                cpu_a_t = np.array(self.runtime.arbitrate(
                    cpu_a_t.tolist(), nav_logits, no_vp_left, stepk, self.max_len))
            else:
                raise NotImplementedError

            # make equiv action
            env_actions, commit_actions, target_pos = [], [], []
            use_tryout = (self.config.IL.tryout and not self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING)
            for i, gmap in enumerate(self.gmaps):
                if cpu_a_t[i] == 0 or stepk == self.max_len - 1 or no_vp_left[i]:
                    vp_stop_scores = [(vp, stop_score) for vp, stop_score in gmap.node_stop_scores.items()]
                    stop_scores = [s[1] for s in vp_stop_scores]
                    stop_vp = vp_stop_scores[np.argmax(stop_scores)][0]
                    stop_pos = gmap.node_pos[stop_vp]
                    if self.config.IL.back_algo == 'control':
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][stop_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    vis_info = {
                        'nodes': list(gmap.node_pos.values()),
                        'ghosts': list(gmap.ghost_aug_pos.values()),
                        'predict_ghost': stop_pos,
                    }
                    env_actions.append({
                        'action': {
                            'act': 0,
                            'cur_vp': cur_vp[i],
                            'stop_vp': stop_vp, 'stop_pos': stop_pos,
                            'back_path': back_path,
                            'tryout': use_tryout,
                        },
                        'vis_info': vis_info,
                    })
                    commit_actions.append(0)
                    target_pos.append(None)
                else:
                    ghost_vp = nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]]
                    ghost_pos = gmap.ghost_aug_pos[ghost_vp]
                    _, front_vp = gmap.front_to_ghost_dist(ghost_vp)
                    front_pos = gmap.node_pos[front_vp]
                    if self.config.VIDEO_OPTION:
                        teacher_action_cpu = teacher_actions[i].cpu().item()
                        if teacher_action_cpu in [0, -100]:
                            teacher_ghost = None
                        else:
                            teacher_ghost = gmap.ghost_aug_pos[nav_inputs['gmap_vp_ids'][i][teacher_action_cpu]]
                        vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            'ghosts': list(gmap.ghost_aug_pos.values()),
                            'predict_ghost': ghost_pos,
                            'teacher_ghost': teacher_ghost,
                        }
                    else:
                        vis_info = None
                    if self.config.IL.back_algo == 'control':
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][front_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    env_actions.append({
                        'action': {
                            'act': 4,
                            'cur_vp': cur_vp[i],
                            'front_vp': front_vp, 'front_pos': front_pos,
                            'ghost_vp': ghost_vp, 'ghost_pos': ghost_pos,
                            'back_path': back_path,
                            'tryout': use_tryout,
                        },
                        'vis_info': vis_info,
                    })
                    commit_actions.append(int(cpu_a_t[i]))
                    target_pos.append(np.array(ghost_pos))
                    prev_vp[i] = front_vp
                    if self.config.MODEL.consume_ghost:
                        gmap.delete_ghost(ghost_vp)
            self.runtime.commit(commit_actions, target_pos, oracle_dist=oracle_dist)

            outputs = self.envs.step(env_actions)
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            if mode == 'eval':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float)
                    pred_path = np.array(info['position']['position'])
                    distances = np.array(info['position']['distance'])
                    metric = {}
                    metric['steps_taken'] = info['steps_taken']
                    metric['distance_to_goal'] = distances[-1]
                    metric['success'] = 1. if distances[-1] <= 3. else 0.
                    metric['oracle_success'] = 1. if (distances <= 3.).any() else 0.
                    metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1], axis=1).sum())
                    metric['collisions'] = info['collisions']['count'] / len(pred_path)
                    gt_length = distances[0]
                    metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
                    dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
                    metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
                    metric['sdtw'] = metric['ndtw'] * metric['success']
                    metric['ghost_cnt'] = self.gmaps[i].ghost_cnt
                    st = self.runtime.states[i]
                    metric['wm_revisions'] = st.revisions
                    metric['wm_stop_vetoes'] = st.vetoes
                    self.stat_eps[ep_id] = metric
                    self.pbar.update()

            if mode == 'infer':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    self.path_eps[ep_id] = [{
                        'position': info['position_infer']['position'][0],
                        'heading': info['position_infer']['heading'][0],
                        'stop': False,
                    }]
                    for p, h in zip(info['position_infer']['position'][1:], info['position_infer']['heading'][1:]):
                        if p != self.path_eps[ep_id][-1]['position']:
                            self.path_eps[ep_id].append({'position': p, 'heading': h, 'stop': False})
                    self.path_eps[ep_id] = self.path_eps[ep_id][:500]
                    self.path_eps[ep_id][-1]['stop'] = True
                    self.pbar.update()

            if sum(dones) > 0:
                for i in reversed(list(range(self.envs.num_envs))):
                    if dones[i]:
                        not_done_index.pop(i)
                        self.envs.pause_at(i)
                        observations.pop(i)
                        self.gmaps.pop(i)
                        prev_vp.pop(i)
                        st = self.runtime.pop(i)
                        if mode == 'eval':
                            self.dev_records.extend(st.records)

            if self.envs.num_envs == 0:
                break

            observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if mode == 'train':
            loss = ml_weight * loss / total_actions
            self.loss += loss
            self.logs['IL_loss'].append(loss.item())
            for k, v in loss_terms.items():
                self.logs[k].append(v / total_actions)
