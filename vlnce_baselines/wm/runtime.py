"""Closed-loop runtime: Imagine -> Execute -> Verify -> Correct.

One call of `ClosedLoopRuntime.step` per navigation step, after ETPNav has
encoded the panorama and scored its topological map:

  1. Verify   the transition committed at the previous step: compare the
              imagined state with the real observation, compute the
              deviation probability, update the belief (RSSM posterior).
  2. Memorise write the verified transition into the corrective memory,
              gated by its reliability; flag the node verified / suspect.
  3. Correct  decide whether to revise (single-step trigger or periodic
              route review over a window, with a revision budget and a
              cooldown, as in NavGPT-3's interrupt-driven runtime); damp
              the stop score of a suspect node.
  4. Imagine  read the memory, imagine every slot of the map (forward WM),
              infer the action towards the imagined target (inverse WM),
              estimate executability, score revised targets (corrector).
  5. Fuse     combine everything with ETPNav's logits.

`commit` then stores the prediction for the action that is actually executed;
`arbitrate` lets the runtime veto a STOP issued from a suspect node.
Executed actions are never rewritten: corrections only act on future
decisions.
"""

import numpy as np
import torch
import torch.nn.functional as F

from vlnce_baselines.wm.forward_wm import gaussian_kl
from vlnce_baselines.wm.inverse_wm import POSE_DIMS
from vlnce_baselines.wm.memory_agent import NUM_VERIFY_FEATS, NUM_SLOT_FLAGS
from vlnce_baselines.wm.fusion import etp_confidence_feats

VERIFIED, SUSPECT = 'verified', 'suspect'
EXEC_SURE = 10.0


class EpisodeState(object):
    """Everything the loop remembers about one running episode."""

    def __init__(self):
        self.h = None                 # RSSM deterministic belief
        self.s = None                 # RSSM stochastic belief
        self.mem_keys = []            # latent state of the stored node
        self.mem_values = []          # U_psi output
        self.mem_rel = []             # reliability of each slot (float)
        self.mem_pos = []             # position of each slot (np.array)
        self.node_flag = {}           # node vp -> VERIFIED / SUSPECT
        self.alt_value = {}           # ghost vp -> best imagined value seen
        self.gap_hist = []
        self.pdev_hist = []
        self.streak = 0
        self.revisions = 0
        self.vetoes = 0
        self.cooldown = 0
        self.prev_dev = 0.0
        self.revise_strength = 0.0
        self.pending = None           # prediction of the committed action
        self.records = []             # (p_dev, label, revised) for analysis


class ClosedLoopRuntime(object):
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.states = []
        self.cache = None
        self._dev_weights = {}

    # ----------------------------------------------------------- bookkeeping
    def reset(self, num_envs):
        self.states = [EpisodeState() for _ in range(num_envs)]
        self.cache = None
        self._dev_weights = {}

    def pop(self, i):
        return self.states.pop(i)

    # ------------------------------------------------------------------ step
    def step(self, pano_feat, txt_embeds, txt_masks,
             gmap_img_fts, gmap_embeds, gmap_pos_fts, valid, etp_logits,
             gmaps, gmap_vp_ids, cur_vp, cur_pos, stepk, max_len,
             teacher_actions=None, oracle_dist=None, train=False):
        cfg, model = self.cfg, self.model
        fwd = model.forward_wm
        B = pano_feat.size(0)
        assert B == len(self.states)
        losses = {}
        self._dev_weights = {}

        pano_feat = pano_feat.float()
        z = model.state_enc(pano_feat)
        for i, st in enumerate(self.states):
            if st.h is None:
                st.h, st.s = [x.squeeze(0) for x in fwd.init_belief(z[i:i + 1])]

        conf = etp_confidence_feats(etp_logits.float(), valid)
        stop_prob = F.softmax(etp_logits.float().masked_fill(~valid, -float('inf')), -1)[:, 0]

        # 1-3. verify, memorise, correct --------------------------------
        stop_scale = [1.0] * B
        infos = [dict() for _ in range(B)]
        ver_idx = [i for i in range(B) if self.states[i].pending is not None]
        if len(ver_idx) > 0:
            ver_losses = self._verify(
                ver_idx, z, pano_feat, conf, stop_prob, gmaps, cur_vp, cur_pos,
                stepk, max_len, oracle_dist, train, stop_scale, infos)
            losses.update(ver_losses)

        # 4. imagine ------------------------------------------------------
        h = torch.stack([st.h for st in self.states], 0)
        s = torch.stack([st.s for st in self.states], 0)
        c, _ = model.reader(torch.cat([h, z], -1), txt_embeds.float(), txt_masks)
        m = self._read_memory(z, c)
        w = model.waypoint_enc(gmap_img_fts.float(), gmap_embeds.float(), gmap_pos_fts.float())
        im = fwd.imagine(h, s, z, w, m, c, valid)

        slot_flags = self._slot_flags(gmaps, gmap_vp_ids, cur_vp, valid)
        corr_logits = model.memory_agent.correct(w, im['z_hat'], m, slot_flags, valid)
        rs = z.new_tensor([st.revise_strength for st in self.states])
        prev_dev = z.new_tensor([st.prev_dev for st in self.states])

        # target for the inverse agent: the forward target, shifted towards
        # the memory agent's revised target while a revision is active
        pi_f = F.softmax(im['value'], -1).detach()
        tgt_logits = im['value']
        if cfg.use_correction:
            corr_logp = F.log_softmax(corr_logits, -1).masked_fill(~valid, 0.)
            tgt_logits = tgt_logits + rs.unsqueeze(-1) * corr_logp
        pi_t = F.softmax(tgt_logits, -1).detach()
        z_tgt = (pi_t.unsqueeze(-1) * im['z_hat']).sum(1)
        inv_logits, _ = model.inverse_wm(z, z_tgt, w, c, valid)
        exec_logits = model.inverse_wm.executability(z, w, im['z_hat'])
        exec_logits = torch.cat([exec_logits.new_full((B, 1), EXEC_SURE), exec_logits[:, 1:]], 1)

        # 5. fuse ----------------------------------------------------------
        use = (cfg.use_forward, cfg.use_inverse, cfg.use_exec, cfg.use_correction)
        fused, lam = model.fusion(etp_logits.float(), im['value'], inv_logits, exec_logits,
                                  corr_logits, rs, prev_dev, valid, use=use)

        # remember imagined alternatives (consulted by later corrections)
        pi_f_np = pi_f.cpu().numpy()
        for i, st in enumerate(self.states):
            for k, vp in enumerate(gmap_vp_ids[i]):
                if k > 0 and vp is not None and vp.startswith('g') and valid[i, k]:
                    st.alt_value[vp] = max(st.alt_value.get(vp, 0.), float(pi_f_np[i, k]))

        if train and teacher_actions is not None:
            losses.update(self._decision_losses(
                teacher_actions, im, z, w, c, valid, fused, corr_logits))

        self.cache = {
            'z': z, 'w': w, 'c': c, 'valid': valid, 'h_k': im['h_k'],
            'prior_mean': im['prior_mean'], 'prior_std': im['prior_std'],
            'z_hat': im['z_hat'], 'exec_logits': exec_logits,
            'gmap_pos_fts': gmap_pos_fts.float(),
        }
        for i, st in enumerate(self.states):
            infos[i]['lambda'] = lam[i].detach().cpu().numpy().tolist()
            infos[i]['revise_strength'] = st.revise_strength
        return {
            'fused_logits': fused,
            'forward_logits': im['value'],
            'inverse_logits': inv_logits,
            'losses': losses,
            'stop_scale': stop_scale,
            'infos': infos,
        }

    # ---------------------------------------------------------------- commit
    def commit(self, actions, target_pos, oracle_dist=None):
        """Store the prediction for the slot each env actually executes.
        actions: list of slot indices (0 = STOP, no transition to verify);
        target_pos: ghost position per env (None for STOP)."""
        cc = self.cache
        for i, st in enumerate(self.states):
            a = int(actions[i])
            if a <= 0 or target_pos[i] is None:
                st.pending = None
                continue
            n = cc['valid'].size(1)
            st.pending = {
                'a_idx': a,
                'h_a': cc['h_k'][i, a],
                'prior_mean': cc['prior_mean'][i, a],
                'prior_std': cc['prior_std'][i, a],
                'z_hat': cc['z_hat'][i, a],
                'z_prev': cc['z'][i],
                'c_prev': cc['c'][i],
                'w_prev': cc['w'][i, :n],
                'w_a': cc['w'][i, a],
                'mask_prev': cc['valid'][i, :n],
                'pose_target': cc['gmap_pos_fts'][i, a, POSE_DIMS],
                'exec_logit': cc['exec_logits'][i, a],
                'target_pos': np.array(target_pos[i], dtype=np.float32),
                'd_goal': None if oracle_dist is None else oracle_dist[i],
            }

    def arbitrate(self, actions, fused_logits, no_vp_left, stepk, max_len):
        """Stop veto: a STOP issued while the memory agent believes the agent
        has deviated is replaced by the best non-STOP slot, within budget."""
        cfg = self.cfg
        if not cfg.stop_veto:
            return actions
        actions = list(actions)
        for i, st in enumerate(self.states):
            if actions[i] != 0 or no_vp_left[i] or stepk >= max_len - 1:
                continue
            if st.prev_dev > cfg.tau_veto and st.vetoes < cfg.veto_budget:
                alt = fused_logits[i, 1:]
                if torch.isfinite(alt).any():
                    actions[i] = int(alt.argmax().item()) + 1
                    st.vetoes += 1
        return actions

    # --------------------------------------------------------------- helpers
    def _read_memory(self, z, c):
        B, d = z.shape
        if not self.cfg.use_memory:
            return z.new_zeros(B, d)
        S = max(1, max(len(st.mem_keys) for st in self.states))
        keys = z.new_zeros(B, S, d)
        values = z.new_zeros(B, S, d)
        log_rel = z.new_zeros(B, S)
        mask = torch.zeros(B, S, dtype=torch.bool, device=z.device)
        for i, st in enumerate(self.states):
            n = len(st.mem_keys)
            if n == 0:
                continue
            keys[i, :n] = torch.stack(st.mem_keys, 0)
            values[i, :n] = torch.stack(st.mem_values, 0)
            log_rel[i, :n] = torch.log(z.new_tensor(st.mem_rel).clamp(min=1e-3))
            mask[i, :n] = True
        m, _ = self.model.memory_agent.memory.read(z, c, keys, values, log_rel, mask)
        return m

    def _slot_flags(self, gmaps, gmap_vp_ids, cur_vp, valid):
        B, N = valid.shape
        flags = np.zeros((B, N, NUM_SLOT_FLAGS), dtype=np.float32)
        for i, st in enumerate(self.states):
            cur_flag = st.node_flag.get(cur_vp[i])
            flags[i, 0] = [cur_flag == SUSPECT, cur_flag == VERIFIED, 0.]
            for k, vp in enumerate(gmap_vp_ids[i]):
                if k == 0 or vp is None or not vp.startswith('g'):
                    continue
                fronts = gmaps[i].ghost_fronts.get(vp, [])
                if len(fronts) > 0:
                    fl = [st.node_flag.get(f) for f in fronts]
                    flags[i, k, 0] = np.mean([f == SUSPECT for f in fl])
                    flags[i, k, 1] = np.mean([f == VERIFIED for f in fl])
                flags[i, k, 2] = st.alt_value.get(vp, 0.)
        return valid.new_tensor(flags, dtype=torch.float)

    def _verify(self, ver_idx, z, pano_feat, conf, stop_prob, gmaps, cur_vp,
                cur_pos, stepk, max_len, oracle_dist, train, stop_scale, infos):
        cfg, model = self.cfg, self.model
        fwd, inv, mem_agent = model.forward_wm, model.inverse_wm, model.memory_agent
        P = [self.states[i].pending for i in ver_idx]
        V = len(ver_idx)
        idx = torch.tensor(ver_idx, device=z.device)
        stack = lambda key: torch.stack([p[key] for p in P], 0)

        h_a, pm, ps, z_hat = stack('h_a'), stack('prior_mean'), stack('prior_std'), stack('z_hat')
        z_prev, c_prev, w_a = stack('z_prev'), stack('c_prev'), stack('w_a')
        z_obs = z[idx]

        s_post, qm, qs = fwd.observe(h_a, z_obs, sample=train)
        kl = gaussian_kl(qm, qs, pm, ps)
        nll = (0.5 * ((qm - pm) / ps) ** 2 + torch.log(ps)).mean(-1)
        gap = 1. - F.cosine_similarity(z_hat, z_obs, dim=-1)

        # hindsight inverse dynamics on the previous slot set
        n_max = max(p['w_prev'].size(0) for p in P)
        d = z.size(1)
        w_prev = z.new_zeros(V, n_max, d)
        mask_prev = torch.zeros(V, n_max, dtype=torch.bool, device=z.device)
        for j, p in enumerate(P):
            n = p['w_prev'].size(0)
            w_prev[j, :n] = p['w_prev']
            mask_prev[j, :n] = p['mask_prev']
        a_idx = torch.tensor([p['a_idx'] for p in P], device=z.device)
        inv_h_logits, pose_h = inv(z_prev, z_obs, w_prev, c_prev, mask_prev)
        inv_nll = F.cross_entropy(inv_h_logits, a_idx, reduction='none')

        pose_err = np.array([
            float(np.linalg.norm(np.array(cur_pos[i])[[0, 2]] - p['target_pos'][[0, 2]]))
            for i, p in zip(ver_idx, P)], dtype=np.float32)
        value_drop = torch.tanh(fwd.state_value(z_hat, c_prev) - fwd.state_value(z_obs, c_prev))

        mem_sim, gap_z, streak = [], [], []
        gap_np = gap.detach().cpu().numpy()
        for j, i in enumerate(ver_idx):
            st = self.states[i]
            if len(st.mem_keys) > 0:
                keys = torch.stack(st.mem_keys, 0)
                mem_sim.append(F.cosine_similarity(keys, z_obs[j:j + 1], dim=-1).max().item())
            else:
                mem_sim.append(0.)
            if len(st.gap_hist) >= 2:
                gap_z.append((gap_np[j] - np.mean(st.gap_hist)) / (np.std(st.gap_hist) + 1e-3))
            else:
                gap_z.append(0.)
            streak.append(min(st.streak, 3) / 3.)

        exec_logit = stack('exec_logit')
        feats = torch.stack([
            gap, nll.clamp(-5, 20), (kl / kl.new_tensor(float(fwd.stoch_dim))).clamp(max=20),
            inv_nll.clamp(max=20), z.new_tensor(np.minimum(pose_err / 3., 5.)), value_drop,
            ps.mean(-1), z.new_tensor(mem_sim), z.new_tensor(np.clip(gap_z, -5, 5)),
            stop_prob[idx], conf[idx, 0],
            z.new_full((V,), float(stepk) / max(max_len, 1)), z.new_tensor(streak),
            torch.sigmoid(exec_logit),
        ], -1)
        assert feats.size(1) == NUM_VERIFY_FEATS
        # verification features are evidence, not targets: no gradient path
        # from the deviation loss back into the world models
        dev_logit = mem_agent.verify(feats.detach(), z_hat.detach(), z_obs.detach())
        p_dev = torch.sigmoid(dev_logit)

        # memory write U_psi, gated by reliability
        rel = (1. - p_dev).detach().clamp(cfg.min_reliability, 1.)
        prev_vals, slots = [], []
        for j, i in enumerate(ver_idx):
            st = self.states[i]
            slot = self._match_slot(st, cur_pos[i])
            slots.append(slot)
            prev_vals.append(st.mem_values[slot] if slot is not None else z.new_zeros(d))
        prev_vals = torch.stack(prev_vals, 0)
        new_vals = mem_agent.memory.write(prev_vals, z_hat, z_obs, w_a, c_prev, feats)
        gate = torch.where(rel >= cfg.write_threshold, rel, rel * cfg.quarantine_scale)
        stored = gate.unsqueeze(-1) * new_vals + (1. - gate.unsqueeze(-1)) * prev_vals

        # labels from the oracle (training supervision / offline analysis only)
        labels = [None] * V
        if oracle_dist is not None:
            for j, (i, p) in enumerate(zip(ver_idx, P)):
                if p['d_goal'] is not None and oracle_dist[i] is not None:
                    labels[j] = float(oracle_dist[i] - p['d_goal'] > cfg.dev_margin)

        p_dev_np = p_dev.detach().cpu().numpy()
        rel_np = rel.cpu().numpy()
        for j, i in enumerate(ver_idx):
            st = self.states[i]
            pd = float(p_dev_np[j])
            # belief update (RSSM posterior)
            st.h, st.s = h_a[j], s_post[j]
            # memory: refresh the slot of a revisited place, else append
            if slots[j] is None:
                st.mem_keys.append(z_obs[j])
                st.mem_values.append(stored[j])
                st.mem_rel.append(float(rel_np[j]))
                st.mem_pos.append(np.array(cur_pos[i], dtype=np.float32))
            else:
                st.mem_keys[slots[j]] = z_obs[j]
                st.mem_values[slots[j]] = stored[j]
                st.mem_rel[slots[j]] = max(st.mem_rel[slots[j]] * cfg.rel_decay, float(rel_np[j]))
            if len(st.mem_keys) > cfg.memory_size:
                drop = int(np.argmin(st.mem_rel[:-1]))
                for lst in (st.mem_keys, st.mem_values, st.mem_rel, st.mem_pos):
                    lst.pop(drop)
            st.gap_hist.append(float(gap_np[j]))
            st.pdev_hist.append(pd)

            # node flags & revision decision
            if pd > cfg.tau_revise:
                st.node_flag[cur_vp[i]] = SUSPECT
                st.streak += 1
            else:
                st.streak = 0
                if pd < cfg.tau_verify:
                    st.node_flag[cur_vp[i]] = VERIFIED
            trigger = pd > cfg.tau_revise
            if cfg.review_every > 0 and (stepk + 1) % cfg.review_every == 0:
                window = st.pdev_hist[-cfg.review_window:]
                trigger = trigger or float(np.mean(window)) > cfg.tau_review
            revised = False
            if trigger and st.revisions < cfg.revision_budget and st.cooldown == 0:
                st.revise_strength = pd
                st.revisions += 1
                st.cooldown = cfg.revision_cooldown
                revised = True
            else:
                st.revise_strength = 0.
                st.cooldown = max(0, st.cooldown - 1)
            if st.node_flag.get(cur_vp[i]) == SUSPECT:
                stop_scale[i] = 1. - pd
            st.prev_dev = pd
            st.pending = None
            st.records.append((pd, labels[j], revised))
            infos[i].update({'p_dev': pd, 'dev_label': labels[j], 'revised': revised,
                             'pose_error': float(pose_err[j])})

        if not train:
            return {}

        losses = {}
        losses['wm_kl'] = F.relu(kl - cfg.free_nats).sum()
        rec = fwd.decode_feat(h_a, s_post)
        tgt = pano_feat[idx].detach()
        losses['wm_rec'] = ((1. - F.cosine_similarity(rec, tgt, dim=-1)) +
                            F.mse_loss(rec, tgt, reduction='none').mean(-1)).sum()
        # open-loop imagination accuracy against a stop-gradient target
        losses['wm_pred'] = (1. - F.cosine_similarity(z_hat, z_obs.detach(), dim=-1)).sum()
        losses['inv_hind'] = inv_nll.sum() + F.mse_loss(
            pose_h, stack('pose_target'), reduction='none').sum(-1).sum()
        arrived = z.new_tensor((pose_err < cfg.exec_threshold).astype(np.float32))
        losses['exec'] = F.binary_cross_entropy_with_logits(exec_logit, arrived, reduction='sum')
        known = [j for j in range(V) if labels[j] is not None]
        if len(known) > 0:
            kidx = torch.tensor(known, device=z.device)
            y = z.new_tensor([labels[j] for j in known])
            pos_w = z.new_tensor(cfg.dev_pos_weight)
            losses['dev'] = F.binary_cross_entropy_with_logits(
                dev_logit[kidx], y, pos_weight=pos_w, reduction='sum')
            self._dev_weights = {ver_idx[j]: labels[j] for j in known}
        return losses

    def _match_slot(self, st, pos):
        if len(st.mem_pos) == 0:
            return None
        pos = np.array(pos, dtype=np.float32)
        dists = [np.linalg.norm(pos[[0, 2]] - p[[0, 2]]) for p in st.mem_pos]
        j = int(np.argmin(dists))
        return j if dists[j] < self.cfg.merge_radius else None

    def _decision_losses(self, teacher, im, z, w, c, valid, fused, corr_logits):
        inv = self.model.inverse_wm
        B, N = valid.shape
        teacher = teacher.long()
        known = teacher != -100
        losses = {}
        if known.sum() == 0:
            return losses
        t = teacher.clamp(min=0)
        kidx = known.nonzero().squeeze(-1)

        losses['nav'] = F.cross_entropy(fused[kidx], t[kidx], reduction='sum')
        losses['fwd_value'] = F.cross_entropy(im['value'][kidx], t[kidx], reduction='sum')

        # goal-conditioned inverse: reach the imagined state of the expert slot
        z_goal = im['z_hat'][torch.arange(B, device=z.device), t]
        g_logits, _ = inv(z, z_goal, w, c, valid)
        losses['inv_goal'] = F.cross_entropy(g_logits[kidx], t[kidx], reduction='sum')

        # cycle consistency: imagined states must be action-discriminable
        cand = valid.clone()
        cand[:, 0] = False
        has = cand.any(-1)
        if has.any():
            probs = cand.float() + (~has).float().unsqueeze(-1) * F.one_hot(
                torch.zeros(B, dtype=torch.long, device=z.device), N).float()
            k = torch.multinomial(probs, 1).squeeze(-1)
            z_k = im['z_hat'][torch.arange(B, device=z.device), k]
            cyc_logits, _ = inv(z, z_k, w, c, valid)
            hidx = has.nonzero().squeeze(-1)
            losses['inv_cycle'] = F.cross_entropy(cyc_logits[hidx], k[hidx], reduction='sum')

        # corrector: revised target = expert slot, emphasised after deviations
        weights = z.new_tensor([
            self.cfg.corr_dev_weight if st.prev_dev > self.cfg.tau_revise or
            self._dev_weights.get(i) == 1. else 1.
            for i, st in enumerate(self.states)])
        ce = F.cross_entropy(corr_logits[kidx], t[kidx], reduction='none')
        losses['corr'] = (ce * weights[kidx]).sum()
        return losses
