# IEVC-Nav：基于双向世界模型与纠偏记忆的闭环 VLN-CE

> 想象（Imagine）→ 执行（Execute）→ 验证（Verify）→ 纠偏（Correct)
>
> 本文档说明在 ETPNav 代码库上实现的「前向世界模型 + 逆向世界模型 + 记忆反馈纠偏」三模块闭环：调研依据、方法设计、训练与实验方案、每个文件的作用和运行方式，以及当前的验证状态。

---

## 0. 一页摘要

| 项目 | 内容 |
|---|---|
| 主干 | ETPNav（TPAMI 2024），**冻结**，不改动其拓扑建图、路点预测和低层控制 |
| 新增模块 | 前向想象智能体 F_θ、逆向动作智能体 G_φ、记忆智能体 U_ψ（验证器 + 纠偏器 + 纠偏记忆库）、不确定性门控融合、闭环运行时 |
| 可训练参数 | 约 **5.6 M**（默认维度 256），ETPNav 主干参数全部冻结 |
| 训练器 | `SS-ETP-WM`（`vlnce_baselines/ss_trainer_ETP_WM.py`） |
| 入口脚本 | `bash run_r2r/main_wm.bash {train|eval|infer} <port> [full|fwd|bidir|bidir_mem|mem_only]` |
| 新增评测指标 | 偏航检测 AUROC / AUPRC / 误报率、每回合纠偏次数、停止否决次数 |
| 验证状态 | 单元测试 + 去掉 Habitat 的集成冒烟测试全部通过；**尚未在 Habitat 中训练和评测，目前没有任何 SR/SPL 结果** |

---

## 1. 调研与定位

### 1.1 现有水平（R2R-CE val-unseen）

| 方法 | 类型 | SR | SPL | 来源 |
|---|---|---|---|---|
| ETPNav | 拓扑规划，全景 | 57.0 | 49.0 | 论文；本地复现 56.99 / 49.01 |
| NavMorph + ETPNav | RSSM 世界模型 + 记忆 | 59.0 | 50.0 | NavMorph（ICCV 2025） |
| NavGPT VLA 8B | 8B VLA，单策略 | 76.94 | 70.61 | NavGPT-3 仓库 README |
| NavGPT-3（GPT-6 Astra planner） | LLM 规划器 + VLA + 运行时 | **81.51** | 70.42 | NavGPT-3 仓库 README |

NavGPT-3 的 arXiv 摘要页和 HTML 版本给出的单独 VLA 数字不一致（74.51 vs 76.94），表中采用仓库 README 的数字，引用前请以最新版本为准。

**定位要实话实说**：用冻结的 ETPNav 主干加 5.6M 参数的插件，不可能超过 NavGPT-3 的 81.51 SR。后者用了 8B VLA、1928 万条训练样本，外加一个前沿 LLM 规划器。所以本项目的 SOTA 目标分两层：

1. **同主干的世界模型 VLN 方法中做到最好**：在 ETPNav 主干上超过 NavMorph+ETPNav（59.0 SR / 50.0 SPL），并在 RxR-CE 上同样验证。这是投稿可以主张的范围。
2. **方法贡献与主干无关**：闭环只用到每一步的「当前状态嵌入 + 候选动作集合 + 主干 logits」三样东西，可以原样套到更强的主干上（见第 7 节）。只有把闭环接到当前最强的主干上，才谈得上刷新整体 SOTA。

### 1.2 参考工作与借鉴点

| 工作 | 借鉴到本实现的部分 |
|---|---|
| **ETPNav** | 冻结主干；拓扑图的 ghost 节点就是前向模型的候选「路点先验」w_t；用 `back_path` 回到已访问节点，作为纠偏的执行手段 |
| **NavMorph / RSSM（Hafner 2019）** | 前向模型采用 RSSM：确定性 GRU 状态 h + 随机隐变量 s，先验 / 后验 + KL + 重建 |
| **SMWM、FutureNav** | 逆动力学监督让隐空间保留可控因素，同时防止表征坍塌；本实现用了后见逆动力学、目标条件逆动力学、循环一致性三种逆向损失 |
| **NWM、DINO-WM** | 在特征空间而不是像素空间里做预测：直接预测 ETPNav 冻结的 768 维全景嵌入，不生成图像 |
| **WorldMemArena** | 用它评测记忆的四个维度来约束记忆设计：**提取**（只写入经过观测验证的经验）、**新鲜度**（回到同一位置时刷新槽位，不重复追加）、**抗干扰**（低可靠的写入被隔离降权）、**问答/使用**（记忆读出送给前向模型和纠偏器） |
| **NavGPT-3** | 运行时设计：①**路线复查中断**（`review_every`，每 K 步对窗口内的偏航证据做一次复查，对应其 Fig.2(A) route-review interrupt）；②**停止不等于结束**（VLA 发出 stop 后，由规划器决定继续、修复、回访还是结束 → 本实现的停止否决 `arbitrate`）；③**回到已访问节点**的修复动作 → 纠偏器偏向「前沿节点已验证」的 ghost；④**中断有预算和授权** → 纠偏预算 `revision_budget` + 冷却 `revision_cooldown`，防止来回振荡 |
| CorrectNav、GroundControl、自监控智能体 | 偏航检测作为一个独立的、可以量化评测的子问题（AUROC/AUPRC/误报率） |

---

## 2. 方法

### 2.1 记号

| 记号 | 含义 | 代码中的来源 |
|---|---|---|
| o_t | 当前节点的全景嵌入（768 维，ETPNav 冻结） | `avg_pano_embeds` |
| z_t = E(o_t) | 世界模型隐状态 | `StateEncoder` |
| k ∈ {0..N} | 拓扑图槽位：0 = STOP，其余为 ghost（已访问节点被屏蔽） | `gmap_vp_ids` |
| w_t^k | 槽位 k 的路点先验：ghost 视觉特征 + ETPNav 指令感知节点嵌入 + 7 维相对位姿 | `WaypointEncoder` |
| I, c_t | 指令；随状态变化的指令读出 | `InstructionReader` |
| M_t, m_t | 纠偏记忆库；其读出向量 | `CorrectiveMemory` |
| (h_t, s_t) | RSSM 信念状态 | `ForwardImaginationAgent` |

### 2.2 前向想象智能体  ẑ_{t+1} = F_θ(z_t, w_t, I, M_t)

文件：`vlnce_baselines/wm/forward_wm.py`

- 一次前向就**批量想象所有槽位**：h^k = GRU([s_t, w^k, m_t, c_t], h_t)，先验 p(s | h^k)，再解码出 ẑ^k。STOP 槽的想象状态就是 z_t 本身。
- **前向目标分数** v^k = V(ẑ^k, c_t)：每个想象状态满足指令的程度，构成前向策略 π_F。
- **偶然不确定性** u^k = 先验标准差的均值，交给验证器，用来区分「门后、转角这类部分可观测带来的正常意外」和「真的走错了」。
- 执行完、拿到真实观测后，用后验 q(s | h^{a*}, z_{t+1}) 修正信念，这就是信念层面的闭环。

### 2.3 逆向动作智能体  a_t = G_φ(z_t, ẑ_{t+1}, I)

文件：`vlnce_baselines/wm/inverse_wm.py`

- 输入当前状态和**目标状态**，对所有槽位打分（π_G），并回归相对位移（sin/cos 朝向、直线距离）。
- **可执行性头** e^k：预测选择槽位 k 后，低层控制器能否真的到达那个 ghost（障碍物、滑动、tryout 失败都会导致到不了）。标签来自执行后的真实位姿误差，也就是开题报告里说的「目标是否可执行」。
- 推理时的目标状态是软目标 ẑ* = Σ_k π_T(k)·ẑ^k，其中 π_T = softmax(v + ρ_t·log π_C)。没有纠偏时 π_T = π_F；一旦触发纠偏（ρ_t > 0），**目标会被拉向记忆智能体修正后的目标**，这就是「修正后的目标反馈给双向模型」。

### 2.4 记忆智能体：验证与纠偏  M_{t+1} = U_ψ(M_t, ẑ_{t+1}, z_{t+1}, a_t, I)

文件：`vlnce_baselines/wm/memory_agent.py`

**(a) 验证器**：执行后把 14 维证据和 |ẑ − z| 一起送进 MLP，输出偏航概率 p_dev。

| # | 特征 | 物理含义 |
|---|---|---|
| 1 | pred_gap_cos | 预测与观测之间的余弦差距 |
| 2 | pred_nll | 观测在先验分布下的 NLL（高不确定性区域会自动降低这一项） |
| 3 | post_prior_kl | 后验相对先验的 KL |
| 4 | inverse_nll | 后见逆动力学：真实发生的转移是否对应原本打算执行的动作 |
| 5 | pose_error | 到达位置与目标 ghost 之间的距离（只用了 ETPNav 本来就在用的位姿） |
| 6 | value_drop | 预期进度和实际进度之差（语言一致性） |
| 7 | pred_uncertainty | 执行槽位的先验离散度 |
| 8 | memory_similarity | 与记忆中各状态的最大相似度（识别绕圈 / 回访） |
| 9 | gap_zscore | 本回合内预测差距的 z 分数（历史证据） |
| 10–11 | etp_stop_prob、etp_entropy | ETPNav 自身的置信度 |
| 12–14 | step_fraction、suspect_streak、exec_prob | 进度、连续可疑次数、执行前预测的可执行性 |

验证特征**不回传梯度**（`detach`），偏航损失只训练验证器，不会反过来把世界模型的预测「拉向容易检测的方向」。

**(b) 纠偏记忆库**（对应 WorldMemArena 的四个维度）

- 写入：v_new = GRU(MLP([ẑ, z, w_a, c, 证据]), v_old)，再用门控 g 混合：v ← g·v_new + (1−g)·v_old。门控取可靠度 r = 1 − p_dev。
- **提取**：只有可靠度 r ≥ `write_threshold` 的转移才按全强度写入。
- **抗干扰**：低可靠的写入乘上 `quarantine_scale` 隔离；读取时注意力加上 log r 偏置，可疑经验自然被降权。
- **新鲜度**：到达位置与已有槽位的距离小于 `merge_radius`，就刷新该槽位，不追加新槽位。容量满时淘汰可靠度最低的槽位。
- 读取：q = W[z_t, c_t]，对「空槽 + 记忆槽」做注意力，得到 m_t，送给前向模型和纠偏器。

**(c) 纠偏器**：对每个可选槽位打分，得到 π_C。输入是 w^k、ẑ^k、m_t，以及 3 个槽位标记：该 ghost 的前沿节点里被标为可疑的比例、被标为已验证的比例，以及之前在已验证节点上想象过的备选价值（`alt_value`）。所以纠偏的效果自然是「回到最近的已验证节点，换一条当时想象过、但没走的路」。

### 2.5 闭环运行时（NavGPT-3 风格的中断与仲裁）

文件：`vlnce_baselines/wm/runtime.py`

```
step t  (ETPNav 已经编码全景、更新拓扑图、给出 logits)
 ├─ 1 Verify    取出 t−1 步提交的预测 → 后验修正信念 → 14 维证据 → p_dev
 ├─ 2 Memorise  按可靠度门控写入 / 刷新槽位；节点标为 verified 或 suspect
 ├─ 3 Correct   p_dev > τ_revise，或每 review_every 步的窗口复查超过 τ_review
 │               且纠偏预算未用完、不在冷却期 → 修正强度 ρ_t = p_dev
 │               可疑节点的 stop 分数乘以 (1 − p_dev)
 ├─ 4 Imagine   读记忆 → 想象全部槽位 → 纠偏器 → 逆向（目标 = π_T 加权）→ 可执行性
 ├─ 5 Fuse      logits = l_ETP + λ_F·log π_F + λ_G·log π_G + λ_E·log σ(e) + γ·ρ_t·log π_C
 ├─ arbitrate  （仅推理）当前节点可疑却发出 STOP → 在否决预算内改选最优的非 STOP 槽位
 └─ commit      为实际执行的动作保存预测，供 t+1 步验证
```

- 门控权重 λ 由 ETPNav 的熵、最大概率和上一步的 p_dev 产生（`FusionGate`）。ETPNav 越不确定，世界模型的话语权越大。最后一层初始化为 0，所以训练开始时 λ 是一个常数，不会一上来就扰乱主干。
- **已执行的动作永远不会被改写**：纠偏只影响之后的决策。ETPNav 原有的「在 stop 分数最高的节点停下」机制，加上被压低的可疑节点 stop 分数，本身就构成「回到已验证节点再停」的修复动作。
- 全部开关关闭时，融合结果和 ETPNav 原始 logits **完全相等**（`test_fusion_is_identity_when_disabled`），所以 A0 基线是严格对照。

### 2.6 训练目标

一个回合内的所有损失求和，再除以动作总数（与 ETPNav 一致），权重见 `WM.LOSS.*`：

| 损失 | 键 | 监督 |
|---|---|---|
| 融合策略交叉熵 | `nav` | 专家动作（ETPNav 的 SPL / nDTW 专家） |
| 前向目标价值 | `fwd_value` | 专家动作 |
| 目标条件逆向 | `inv_goal` | G(z_t, ẑ^{专家}) → 专家动作 |
| 循环一致性 | `inv_cycle` | G(z_t, ẑ^k) → k，k 随机抽取，保证想象状态可以区分不同动作 |
| 后见逆向 + 位移回归 | `inv_hind` | G(z_t, z_{t+1}) → 实际执行的动作 |
| RSSM KL（free nats） | `wm_kl` | KL(q‖p) |
| 特征重建 | `wm_rec` | 解码结果对齐 ETPNav 的 768 维全景嵌入（cos + MSE） |
| 开环预测 | `wm_pred` | 1 − cos(ẑ^{a*}, sg(z_{t+1})) |
| 可执行性 | `exec` | 实际到达误差 < `exec_threshold` |
| 偏航 | `dev` | 测地距离到目标的增量 > `dev_margin`（`pos_weight` 用于平衡类别） |
| 纠偏目标 | `corr` | 专家动作；偏航之后的步按 `corr_dev_weight` 加权 |

**偏航数据从哪来**（开题报告难点 2）：①ETPNav 训练时本来就按策略采样（`sample_ratio` 衰减），会自然偏离专家；②额外按 `perturb_prob` 的概率把动作替换成随机的非专家 ghost，在 Habitat 里制造偏航和恢复的轨迹。偏航标签只在训练时和离线评测时用到 oracle 距离，**推理决策不使用任何特权信息**。

RxR-CE 的指令沿参考路径而不一定走最短路，用测地距离判定偏航只是近似；如果 RxR 的检测指标不理想，可以改成「到参考路径的距离」（环境里已有 `current_dist_to_refpath`）。

---

## 3. 实验方案（对应开题报告的 A0–A5）

| 阶段 | 命令 | 设置 |
|---|---|---|
| A0 ETPNav 基线 | `bash run_r2r/main.bash eval <port>` | 原版训练器 |
| A1 仅前向世界模型 | `bash run_r2r/main_wm.bash train <port> fwd` | 只开前向 |
| A2 双向世界模型 | `... bidir` | 前向 + 逆向 + 可执行性 |
| A3 + 记忆验证 | `... bidir_mem` | 加记忆读写与偏航检测，不做纠偏和停止否决 |
| 仅历史记忆（对照） | `... mem_only` | 关掉前向、逆向、可执行性三项融合，只保留记忆纠偏 |
| A4 完整闭环 | `... full` | 全部打开 |
| A5 消融 / 效率 | 修改 `WM.*` 开关、`WM.LOSS.*` 权重和阈值 | 另外统计每步推理耗时 |

评测时，每个检查点除了 ETPNav 原有的统计文件，还会额外生成：

- `RESULTS_DIR/<exp>/stats_ep_ckpt_*.json`：每个回合新增 `wm_revisions` 和 `wm_stop_vetoes` 两个字段；
- `RESULTS_DIR/<exp>/wm_deviation_ckpt_*_r{rank}.json`：偏航检测的 `dev_auroc`、`dev_auprc`、`dev_false_alarm`、`dev_rate`、`num_revisions`。

建议的调参顺序：先在 val_seen 上对 `tau_revise`、`tau_veto`、`review_every` 做网格搜索（只需要评测，不用重新训练），再到 val_unseen 上报告结果。开题报告的难点 1（预测误差和偏航难以区分）可以直接用 `wm_deviation` 里的 AUROC 来判断：AUROC 不够时，先改验证器的证据设计，不要继续往上叠模块。

---

## 4. 文件说明

### 新增

| 文件 | 作用 |
|---|---|
| `vlnce_baselines/wm/__init__.py` | 闭环模块包的导出 |
| `vlnce_baselines/wm/latent.py` | `StateEncoder`（o→z）、`WaypointEncoder`（拓扑图槽位→w，含 STOP 嵌入）、`InstructionReader`（随状态变化的指令注意力） |
| `vlnce_baselines/wm/forward_wm.py` | 前向想象智能体（RSSM：信念初始化、全槽位批量想象、后验观测、特征/隐状态解码、价值头）和高斯 KL |
| `vlnce_baselines/wm/inverse_wm.py` | 逆向动作智能体（目标条件槽位打分、位移回归、可执行性头） |
| `vlnce_baselines/wm/memory_agent.py` | 纠偏记忆库（门控写入 / 带可靠度偏置的读取）、验证器、纠偏器；验证特征和槽位标记的定义表 |
| `vlnce_baselines/wm/fusion.py` | 不确定性门控融合；屏蔽槽位安全处理（避免 λ·(−inf) 在反向传播时产生 NaN） |
| `vlnce_baselines/wm/bidir_wm.py` | `BiWorldModelNav`：所有可训练参数的容器，`from_config(config.WM)` |
| `vlnce_baselines/wm/runtime.py` | `ClosedLoopRuntime`：验证 → 记忆 → 纠偏 → 想象 → 融合、`commit`、`arbitrate`（停止否决）、回合状态 `EpisodeState`、全部训练损失 |
| `vlnce_baselines/wm/metrics.py` | 不依赖 sklearn 的 AUROC / AUPRC / 误报率和 `deviation_report` |
| `vlnce_baselines/ss_trainer_ETP_WM.py` | 训练器 `SS-ETP-WM`：加载并冻结 ETPNav、只优化闭环参数（多卡时手动 all-reduce 梯度，不需要 DDP）、改写 rollout 接入闭环、加入扰动、输出偏航评测文件；检查点里只存 `wm_state_dict`（主干解冻时才同时存主干） |
| `run_r2r/main_wm.bash`、`run_rxr/main_wm.bash` | 训练、评测、推理脚本，第三个参数选择消融变体 |
| `tests/test_wm_loop.py` | 不依赖 Habitat 的单元测试：训练闭环反向传播（检查所有子模块都有有限梯度）、评测与消融、纠偏和停止否决路径、全部关闭时融合退化为恒等、指标 |
| `tests/smoke_trainer_mock.py` | 集成冒烟测试：把 Habitat、模拟器和 ETPNav 网络换成假实现，实际运行 `WMTrainer` 的 train、eval、infer 三种 rollout（会全局打补丁，**只能在没有 Habitat 的机器上运行**） |
| `docs/WM_DESIGN.md` | 本文档 |

### 修改

| 文件 | 改动 |
|---|---|
| `vlnce_baselines/__init__.py` | 注册 `ss_trainer_ETP_WM` |
| `vlnce_baselines/config/default.py` | 新增 `_C.WM` 配置节点（主干路径、模型维度、模块开关、阈值、记忆、监督、损失权重），全部可以在命令行用 `WM.xxx` 覆盖 |

ETPNav 原有的训练器、模型和环境代码**一行都没改**，原版 `run_r2r/main.bash` 照常可用。

---

## 5. 运行

```bash
# 0. 先准备好 ETPNav 原版环境、数据，以及微调好的 ETPNav 检查点（A0）
#    脚本默认的 R2R 检查点是 data/logs/checkpoints/release_r2r/ckpt.iter12000.pth
#    RxR 在 run_rxr/main_wm.bash 里把 backbone 指向 release_rxr/ckpt.iter19600.pth

# 1. 训练完整闭环（2 卡，主干冻结）
bash run_r2r/main_wm.bash train 2333 full
# 2. 评测该实验目录下的所有检查点
bash run_r2r/main_wm.bash eval 2333 full
# 3. 生成测试集提交文件
bash run_r2r/main_wm.bash infer 2333 full

# 不依赖 Habitat 的自检（任意装了 PyTorch 的机器）
python -m pytest tests/test_wm_loop.py
python tests/smoke_trainer_mock.py
```

常用覆盖项：`WM.freeze_backbone False`（联合微调主干，学习率用 `IL.lr`）、`WM.perturb_prob 0.2`、`WM.review_every 0`（关闭周期复查）、`WM.log_dev_metrics False`（评测时不调用 oracle 距离）。

---

## 6. 验证状态（请先读这一节再引用结果）

已经做的：
- `tests/test_wm_loop.py` 5 项测试全部通过：反向传播无 NaN，每个子模块都拿到梯度；三种消融配置都能跑；纠偏和停止否决路径确实会触发，且不超预算；全部关闭时融合输出与 ETPNav 逐元素相等。
- `tests/smoke_trainer_mock.py` 通过：真实的 `WMTrainer._train_interval` 和 `rollout('eval' / 'infer')` 能在假环境里跑通，损失有限，能产出偏航报告和推理路径。冒烟测试里的 AUROC 是未训练模型在随机数据上的结果，没有任何意义。

**没有做的**：
- 当前容器里没有 Habitat、MP3D 数据和 GPU，所以**没有在真实环境中训练或评测**，文档里也**没有声称任何 SR/SPL 提升**。
- 代码按 ETPNav 原环境（Python 3.6 / PyTorch 1.9.1）的写法编写，没有用到更高版本的语法或 API，但测试是在 Python 3.13 / PyTorch 2.x 上跑的。第一次在真实环境运行时，建议先用 `NUM_ENVIRONMENTS 1`、`IL.iters 20` 快速跑一遍，确认没有问题。
- 多卡梯度同步（`_allreduce_grads`）没有在多卡环境中实际跑过。

---

## 7. 后续：怎样才可能刷新整体 SOTA

1. **先证明闭环本身有收益**（A1–A5 对比 A0 和 NavMorph+ETPNav），同时报告偏航检测指标。这是论文的核心贡献：「预测与观测的偏差，是比导航置信度更好的错误信号」。
2. **换更强的主干**：闭环对主干只有三个接口要求——当前状态嵌入 o_t、候选动作集合（拓扑图槽位或 VLA 的局部路点）及其特征、主干 logits。以 NavGPT VLA（Qwen3-VL 加动作头，权重已开源）为例：o_t 可以取 VLA 的视觉 token 池化，候选可以取 VLA 预测的局部路点，闭环作为 NavGPT-3 运行时里的一个「监控线程」，用 p_dev 触发它的 route-review 中断。这也和 NavGPT-3 把推理、执行、监控拆成不同线程的设计相符。
3. **阈值在线标定**：用 val_seen 标定 τ，让误报率控制在 5% 以内；也可以把触发规则换成一个学出来的停止 / 修复决策头。
4. **更强的验证证据**：把可执行性和位姿误差扩展到多步（低层控制每一步的碰撞和 tryout 记录），并加入深度一致性。

---

## 参考

- ETPNav：An et al., *ETPNav: Evolving Topological Planning for VLN-CE*, TPAMI 2024.
- NavMorph：Yao et al., ICCV 2025.
- NavGPT-3：Zhou et al., *Harnessing Context in a Hierarchical Navigation Runtime*, arXiv:2610.10787；代码 https://github.com/metacognitionai/NavGPT-3
- WorldMemArena：https://github.com/UCSB-AI/WorldMemArena
- RSSM：Hafner et al., ICML 2019；NWM：Bar et al., CVPR 2025；DINO-WM：Zhou et al., ICML 2025；SMWM：arXiv:2606.20104；FutureNav：arXiv:2606.30367；CorrectNav：arXiv:2508.10416；GroundControl：arXiv:2606.20479。
