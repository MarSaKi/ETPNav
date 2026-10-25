"""Habitat-free tests of the closed-loop world-model modules.

Run: python -m pytest tests/test_wm_loop.py   (or python tests/test_wm_loop.py)
A mock topological map reproduces the tensor layout ETPNav feeds the
runtime: slot 0 = STOP, then visited nodes, then ghosts.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    import habitat  # noqa: F401
except ImportError:
    # the package __init__ pulls in habitat trainers; the WM modules do not
    # need them, so expose the package path without running __init__
    import types
    _pkg = types.ModuleType('vlnce_baselines')
    _pkg.__path__ = [os.path.join(ROOT, 'vlnce_baselines')]
    sys.modules.setdefault('vlnce_baselines', _pkg)

from vlnce_baselines.wm import BiWorldModelNav, ClosedLoopRuntime  # noqa: E402
from vlnce_baselines.wm.metrics import auroc, auprc, deviation_report  # noqa: E402

FEAT = 768


def wm_cfg(**kw):
    cfg = dict(
        latent_dim=64, deter_dim=64, stoch_dim=8, dropout=0.0,
        fusion_init=0.3, correction_gamma=1.0,
        use_forward=True, use_inverse=True, use_exec=True, use_memory=True,
        use_correction=True, stop_veto=True,
        tau_revise=0.5, tau_verify=0.3, tau_review=0.4, tau_veto=0.6,
        review_every=3, review_window=3, revision_budget=3, revision_cooldown=1,
        veto_budget=1, memory_size=8, merge_radius=0.5, min_reliability=0.05,
        write_threshold=0.5, quarantine_scale=0.2, rel_decay=0.9,
        free_nats=0.5, exec_threshold=1.0, dev_margin=0.5, dev_pos_weight=2.0,
        corr_dev_weight=3.0,
    )
    cfg.update(kw)
    return SimpleNamespace(**cfg)


class MockGraph(object):
    def __init__(self, rng):
        self.rng = rng
        self.node_pos = {}
        self.ghost_pos = {}
        self.ghost_fronts = {}
        self.node_stop_scores = {}
        self.cnt = 0

    def arrive(self, pos):
        vp = str(len(self.node_pos))
        self.node_pos[vp] = np.array(pos, dtype=np.float32)
        for _ in range(self.rng.randint(1, 4)):
            g = 'g%d' % self.cnt
            self.cnt += 1
            self.ghost_pos[g] = self.node_pos[vp] + self.rng.uniform(-2, 2, 3).astype(np.float32)
            self.ghost_fronts[g] = [vp]
        return vp

    def vp_ids(self):
        return [None] + list(self.node_pos) + list(self.ghost_pos)


def batch_inputs(gmaps, B, L=12):
    ids = [g.vp_ids() for g in gmaps]
    N = max(len(x) for x in ids)
    valid = torch.zeros(B, N, dtype=torch.bool)
    for i, x in enumerate(ids):
        valid[i, 0] = True
        for k, vp in enumerate(x):
            if vp is not None and vp.startswith('g'):
                valid[i, k] = True
    etp = torch.randn(B, N).masked_fill(~valid, -float('inf'))
    return dict(
        pano_feat=torch.randn(B, FEAT),
        txt_embeds=torch.randn(B, L, FEAT),
        txt_masks=torch.ones(B, L, dtype=torch.bool),
        gmap_img_fts=torch.randn(B, N, FEAT),
        gmap_embeds=torch.randn(B, N, FEAT),
        gmap_pos_fts=torch.randn(B, N, 7),
        valid=valid, etp_logits=etp,
    ), ids


def run_episode(model, cfg, B=3, T=6, train=True, seed=0, finished=None):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    rt = ClosedLoopRuntime(model, cfg)
    rt.reset(B)
    gmaps = [MockGraph(rng) for _ in range(B)]
    pos = [np.zeros(3, dtype=np.float32) for _ in range(B)]
    cur_vp = [g.arrive(p) for g, p in zip(gmaps, pos)]
    dist = [10.0] * B
    total = 0.
    alive = list(range(B))
    for t in range(T):
        inp, ids = batch_inputs(gmaps, len(gmaps))
        teacher = torch.stack([
            torch.multinomial(inp['valid'][i].float(), 1)[0] for i in range(len(gmaps))])
        out = rt.step(gmaps=gmaps, gmap_vp_ids=ids, cur_vp=cur_vp,
                      cur_pos=pos, stepk=t, max_len=T,
                      teacher_actions=teacher if train else None,
                      oracle_dist=dist, train=train, **inp)
        fused = out['fused_logits']
        assert fused.shape == inp['valid'].shape
        assert torch.isfinite(fused[inp['valid']]).all()
        assert torch.isinf(fused[~inp['valid']]).all()
        for v in out['losses'].values():
            assert torch.isfinite(v), out['losses']
            total = total + v
        acts = fused.argmax(-1).tolist()
        no_vp_left = [len(g.ghost_pos) == 0 for g in gmaps]
        acts = rt.arbitrate(acts, fused, no_vp_left, t, T)
        targets = []
        for i, a in enumerate(acts):
            targets.append(gmaps[i].ghost_pos[ids[i][a]] if a > 0 else None)
        rt.commit(acts, targets, oracle_dist=dist)
        # execute: arrive near the ghost (sometimes blocked), new node appears
        done = []
        for i, a in enumerate(acts):
            if a == 0 or t == T - 1:
                done.append(i)
                continue
            g = ids[i][a]
            noise = rng.uniform(-1.5, 1.5, 3).astype(np.float32) if rng.rand() < 0.3 else 0.
            pos[i] = gmaps[i].ghost_pos.pop(g) + noise
            gmaps[i].ghost_fronts.pop(g)
            cur_vp[i] = gmaps[i].arrive(pos[i])
            dist[i] = dist[i] + rng.uniform(-2, 1)
        for i in reversed(done):
            st = rt.pop(i)
            if finished is not None:
                finished.append(st)
            gmaps.pop(i)
            pos.pop(i)
            cur_vp.pop(i)
            dist.pop(i)
            alive.pop(i)
        if len(gmaps) == 0:
            break
    return total


def test_train_loop_backward():
    cfg = wm_cfg()
    model = BiWorldModelNav.from_config(cfg)
    model.train()
    loss = run_episode(model, cfg, train=True)
    assert torch.is_tensor(loss) and torch.isfinite(loss)
    loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters()}
    for n, g in grads.items():
        assert g is None or torch.isfinite(g).all(), 'non-finite grad in ' + n
    for prefix in ['forward_wm.cell', 'inverse_wm.slot_score', 'memory_agent.verifier',
                   'memory_agent.corrector', 'fusion.base', 'state_enc']:
        assert any(g is not None and g.abs().sum() > 0
                   for n, g in grads.items() if n.startswith(prefix)), prefix


def test_eval_loop_and_ablations():
    for kw in [dict(), dict(use_forward=False, use_inverse=False, use_exec=False),
               dict(use_memory=False, use_correction=False, stop_veto=False)]:
        cfg = wm_cfg(**kw)
        model = BiWorldModelNav.from_config(cfg).eval()
        with torch.no_grad():
            run_episode(model, cfg, train=False, seed=1)


def test_revision_and_stop_veto_paths():
    # thresholds at zero: every verified step revises (within budget) and
    # every STOP from a suspect node is vetoed once
    cfg = wm_cfg(tau_revise=0.0, tau_veto=0.0, revision_budget=2, revision_cooldown=0)
    model = BiWorldModelNav.from_config(cfg).eval()
    finished = []
    with torch.no_grad():
        run_episode(model, cfg, B=4, T=8, train=False, seed=2, finished=finished)
    assert len(finished) == 4
    assert any(st.revisions > 0 for st in finished)
    assert all(st.revisions <= 2 for st in finished)
    assert all(len(st.records) == len(st.pdev_hist) for st in finished)
    assert any(st.node_flag for st in finished)


def test_fusion_is_identity_when_disabled():
    cfg = wm_cfg(use_forward=False, use_inverse=False, use_exec=False, use_correction=False)
    model = BiWorldModelNav.from_config(cfg).eval()
    rt = ClosedLoopRuntime(model, cfg)
    rt.reset(2)
    rng = np.random.RandomState(0)
    gmaps = [MockGraph(rng) for _ in range(2)]
    cur_vp = [g.arrive(np.zeros(3)) for g in gmaps]
    inp, ids = batch_inputs(gmaps, 2)
    with torch.no_grad():
        out = rt.step(gmaps=gmaps, gmap_vp_ids=ids, cur_vp=cur_vp,
                      cur_pos=[np.zeros(3)] * 2, stepk=0, max_len=5, **inp)
    v = inp['valid']
    assert torch.allclose(out['fused_logits'][v], inp['etp_logits'][v])


def test_metrics():
    assert abs(auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) - 1.0) < 1e-6
    assert abs(auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) - 0.0) < 1e-6
    assert abs(auprc([0.1, 0.9], [0, 1]) - 1.0) < 1e-6
    rep = deviation_report([(0.9, 1, True), (0.1, 0, False), (0.3, None, False)], 0.5)
    assert rep['num_verified_steps'] == 2 and rep['dev_auroc'] == 1.0


if __name__ == '__main__':
    test_train_loop_backward()
    test_eval_loop_and_ablations()
    test_revision_and_stop_veto_paths()
    test_fusion_is_identity_when_disabled()
    test_metrics()
    print('all tests passed')
