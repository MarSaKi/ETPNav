"""Smoke test: run the real WMTrainer.rollout (train / eval / infer) with
Habitat, the simulator and the ETPNav network replaced by fakes.

Checks the integration of the closed loop with ETPNav's rollout on a machine
without Habitat / GPU:  python tests/smoke_trainer_mock.py
Do not run it inside the real VLN-CE environment: it patches torch and
numpy globally (Tensor.cuda -> no-op, np.float alias) and stubs habitat.
"""
import importlib.abc
import importlib.machinery
import logging
import os
import sys
import types
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
np.float = float
torch.Tensor.cuda = lambda self, *a, **k: self

STUBS = ('habitat', 'habitat_baselines', 'habitat_extensions', 'gym', 'tensorflow',
         'lmdb', 'msgpack_numpy', 'jsonlines', 'fastdtw', 'matplotlib')


class StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        m = MagicMock(name=self.__name__ + '.' + name)
        setattr(self, name, m)
        return m


class StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in STUBS:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        m = StubModule(spec.name)
        m.__path__ = []
        return m

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, StubFinder())


class _Registry(object):
    def register_trainer(self, name=None):
        return lambda cls: cls

    def register_policy(self, cls=None, name=None):
        return cls if cls is not None else (lambda c: c)


class BaseILTrainer(object):
    def __init__(self, config=None):
        self.config = config


import habitat_baselines.common.baseline_registry as br  # noqa: E402
br.baseline_registry = _Registry()
import habitat_baselines.common.base_il_trainer as bil  # noqa: E402
bil.BaseILTrainer = BaseILTrainer
import habitat  # noqa: E402
habitat.logger = logging.getLogger('smoke')
import habitat_baselines.common.obs_transformers as ot  # noqa: E402
ot.apply_obs_transforms_batch = lambda batch, tf: batch
import habitat_baselines.utils.common as huc  # noqa: E402


def batch_obs(observations, device=None):
    return {k: torch.as_tensor(np.stack([o[k] for o in observations])) for k in observations[0]}


huc.batch_obs = batch_obs
import fastdtw  # noqa: E402
fastdtw.fastdtw = lambda a, b, dist=None: (float(np.abs(len(a) - len(b))), None)

pkg = types.ModuleType('vlnce_baselines')
pkg.__path__ = [os.path.join(ROOT, 'vlnce_baselines')]
sys.modules['vlnce_baselines'] = pkg

import vlnce_baselines.models.graph_utils as gu  # noqa: E402
gu.heading_from_quaternion = lambda q: float(q[0]) % (2 * np.pi)
import vlnce_baselines.ss_trainer_ETP_WM as twm  # noqa: E402
twm.GraphMap = gu.GraphMap
from vlnce_baselines.wm import BiWorldModelNav, ClosedLoopRuntime  # noqa: E402

D = 768
rng = np.random.RandomState(0)


class FakeNet(object):
    num_recurrent_layers = 1

    def __call__(self, mode=None, **kw):
        if mode == 'language':
            B, L = kw['txt_ids'].shape
            return torch.randn(B, L, D)
        if mode == 'waypoint':
            B = kw['observations']['rgb'].shape[0]
            out = {k: [] for k in ['cand_rgb', 'cand_depth', 'cand_angle_fts', 'cand_img_idxes',
                                   'cand_angles', 'cand_distances']}
            for _ in range(B):
                K = rng.randint(1, 4)
                idx = rng.choice(12, K, replace=False)
                out['cand_rgb'].append(torch.randn(K, 512))
                out['cand_depth'].append(torch.randn(K, 128))
                out['cand_angle_fts'].append(torch.randn(K, 4))
                out['cand_img_idxes'].append(idx)
                out['cand_angles'].append(list(rng.uniform(0, 2 * np.pi, K)))
                out['cand_distances'].append(list(rng.uniform(0.5, 3, K)))
            out.update({'pano_rgb': torch.randn(B, 12, 512), 'pano_depth': torch.randn(B, 12, 128),
                        'pano_angle_fts': torch.randn(12, 4), 'pano_img_idxes': np.arange(12)})
            return out
        if mode == 'panorama':
            B, V = kw['rgb_fts'].shape[:2]
            masks = torch.arange(V).unsqueeze(0) < kw['view_lens'].unsqueeze(1)
            return torch.randn(B, V, D).half(), masks
        if mode == 'navigation':
            B, N = kw['gmap_masks'].shape
            logits = torch.randn(B, N).half()
            logits.masked_fill_(kw['gmap_visited_masks'], -float('inf'))
            logits.masked_fill_(~kw['gmap_masks'], -float('inf'))
            return {'gmap_embeds': torch.randn(B, N, D).half(), 'global_logits': logits}
        raise ValueError(mode)


class FakePolicy(object):
    net = FakeNet()

    def eval(self):
        return self

    def train(self, mode=True):
        return self

    def parameters(self):
        return iter([])


class FakeEnvs(object):
    def __init__(self, n, n_eps):
        self.all = list(range(n))
        self.active = list(range(n))
        self.n_eps = n_eps
        self.ep_counter = 0
        self.ep = {}
        self.pos = {}
        self.goal = {}
        self.path = {}

    @property
    def num_envs(self):
        return len(self.active)

    def resume_all(self):
        self.active = list(self.all)

    def pause_at(self, i):
        self.active.pop(i)

    def _obs(self):
        return {'instruction': rng.randint(1, 100, 80), 'rgb': np.zeros((4, 4, 3), np.float32),
                'depth': np.zeros((4, 4, 1), np.float32)}

    def reset(self):
        obs = []
        for e in self.active:
            self.ep[e] = NS(episode_id=str(self.ep_counter % self.n_eps),
                            instruction=NS(instruction_id=self.ep_counter))
            self.ep_counter += 1
            self.pos[e] = np.zeros(3, np.float32)
            self.goal[e] = rng.uniform(-8, 8, 3).astype(np.float32)
            self.goal[e][1] = 0
            self.path[e] = [self.pos[e].copy()]
            obs.append(self._obs())
        return obs

    def current_episodes(self):
        return [self.ep[e] for e in self.active]

    def call(self, names, kwargs=None):
        return [self.call_at(i, n) for i, n in enumerate(names)]

    def call_at(self, i, name, kwargs=None):
        e = self.active[i]
        if name == 'get_pos_ori':
            return self.pos[e].copy(), np.array([rng.uniform(0, 6.28), 0, 0, 0])
        if name == 'current_dist_to_goal':
            return float(np.linalg.norm(self.pos[e] - self.goal[e]))
        if name == 'point_dist_to_goal':
            return float(np.linalg.norm(np.array(kwargs['pos']) - self.goal[e]))
        if name == 'get_cand_real_pos':
            ang, dis = kwargs['angle'], kwargs['forward']
            return self.pos[e] + np.array([-dis * np.sin(ang), 0, -dis * np.cos(ang)], np.float32)
        raise ValueError(name)

    def step(self, actions):
        outs = []
        for i, a in enumerate(actions):
            e = self.active[i]
            act = a['action']
            if act['act'] == 4:
                self.pos[e] = np.array(act['ghost_pos'], np.float32) + \
                    (rng.uniform(-1, 1, 3).astype(np.float32) * (rng.rand() < 0.3))
                self.pos[e][1] = 0
                self.path[e].append(self.pos[e].copy())
                outs.append((self._obs(), 0., False, {}))
            else:
                self.pos[e] = np.array(act['stop_pos'], np.float32)
                self.path[e].append(self.pos[e].copy())
                d = [float(np.linalg.norm(p - self.goal[e])) for p in self.path[e]]
                info = {'position': {'position': [p.tolist() for p in self.path[e]], 'distance': d},
                        'position_infer': {'position': [p.tolist() for p in self.path[e]],
                                           'heading': [0.] * len(self.path[e])},
                        'steps_taken': len(self.path[e]), 'collisions': {'count': 0}}
                outs.append((self._obs(), 0., True, info))
        return outs


def wm_cfg():
    c = dict(latent_dim=64, deter_dim=64, stoch_dim=8, dropout=0.0, fusion_init=0.3,
             correction_gamma=1.0, use_forward=True, use_inverse=True, use_exec=True,
             use_memory=True, use_correction=True, stop_veto=True, fused_stop_scores=True,
             tau_revise=0.5, tau_verify=0.3, tau_review=0.4, tau_veto=0.6, review_every=3,
             review_window=3, revision_budget=3, revision_cooldown=1, veto_budget=1,
             memory_size=8, merge_radius=0.5, min_reliability=0.05, write_threshold=0.5,
             quarantine_scale=0.2, rel_decay=0.9, free_nats=0.5, exec_threshold=1.0,
             dev_margin=0.5, dev_pos_weight=2.0, corr_dev_weight=3.0, perturb_prob=0.3,
             log_dev_metrics=True, freeze_backbone=True, backbone_amp=False, grad_clip=5.0)
    c['LOSS'] = NS(nav=1., fwd_value=.5, inv_goal=.5, inv_cycle=.2, inv_hind=.5, wm_kl=.1,
                   wm_rec=.5, wm_pred=1., exec=.2, dev=1., corr=.5)
    return NS(**c)


def make_trainer(task='r2r'):
    t = object.__new__(twm.WMTrainer)
    t.config = NS(
        IL=NS(max_text_len=80, ghost_aug=0., loc_noise=0.5, waypoint_aug=True, tryout=True,
              back_algo='control', expert_policy='spl'),
        MODEL=NS(task_type=task, merge_ghost=True, consume_ghost=True),
        TASK_CONFIG=NS(TASK=NS(INSTRUCTION_SENSOR_UUID='rxr_instruction'),
                       SIMULATOR=NS(HABITAT_SIM_V0=NS(ALLOW_SLIDING=True))),
        VIDEO_OPTION=[],
    )
    t.wm_cfg = wm_cfg()
    t.max_len = 8
    t.device = torch.device('cpu')
    t.obs_transforms = []
    t.policy = FakePolicy()
    t.waypoint_predictor = MagicMock()
    t.wm = BiWorldModelNav.from_config(t.wm_cfg)
    t.runtime = ClosedLoopRuntime(t.wm, t.wm_cfg)
    t.optimizer = torch.optim.AdamW(t.wm.parameters(), lr=1e-4)
    t.world_size, t.local_rank = 1, 0
    t.envs = FakeEnvs(3, n_eps=6)
    t.gt_data = {str(k): {'locations': [[0, 0, 0], [1, 0, 1]]} for k in range(6)}
    t.dev_records = []
    return t


def main():
    t = make_trainer()
    # --- train: two optimisation steps through the real _train_interval
    t._train_interval(2, ml_weight=1.0, sample_ratio=0.5)
    print('train logs:', {k: round(float(np.mean(v)), 3) for k, v in t.logs.items()})
    assert np.isfinite(t.logs['IL_loss']).all()

    # --- eval
    t.wm.eval()
    t.stat_eps = {}
    t.pbar = MagicMock()
    with torch.no_grad():
        while len(t.stat_eps) < 6:
            t.rollout('eval')
    print('eval episodes:', len(t.stat_eps), 'example:', list(t.stat_eps.values())[0])
    from vlnce_baselines.wm.metrics import deviation_report  # noqa: E402
    print('deviation report:', deviation_report(t.dev_records, 0.5))

    # --- infer
    t.path_eps, t.inst_ids = {}, {}
    with torch.no_grad():
        while len(t.path_eps) < 6:
            t.rollout('infer')
    print('inferred paths:', len(t.path_eps))
    print('SMOKE OK')


if __name__ == '__main__':
    main()
