# ETPNav for VLN-CE

Code of our paper "ETPNav: Evolving Topological Planning for Vision-Language Navigation in Continuous Environments" [[Paper]](https://arxiv.org/pdf/2304.03047v2.pdf)

🔥Winner of the [RxR-Habitat Challenge](https://embodied-ai.org/cvpr2022) in CVPR 2022. [[Challenge Report]](https://arxiv.org/abs/2206.11610) [[Challenge Certificate]](https://marsaki.github.io/assets/cert/rxr-habitat-cert.pdf)

Vision-language navigation is a task that requires an agent to follow instructions to navigate in environments. It becomes increasingly crucial in the field of embodied AI, with potential applications in autonomous navigation, search and rescue, and human-robot interaction. In this paper, we propose to address a more practical yet challenging counterpart setting - vision-language navigation in continuous environments (VLN-CE). To develop a robust VLN-CE agent, we propose a new navigation framework, ETPNav, which focuses on two critical skills: 1) the capability to abstract environments and generate long-range navigation plans, and 2) the ability of obstacle-avoiding control in continuous environments. ETPNav performs online topological mapping of environments by self-organizing predicted waypoints along a traversed path, without prior environmental experience. It privileges the agent to break down the navigation procedure into high-level planning and low-level control. Concurrently, ETPNav utilizes a transformer-based cross-modal planner to generate navigation plans based on topological maps and instructions. The plan is then performed through an obstacle-avoiding controller that leverages a trial-and-error heuristic to prevent navigation from getting stuck in obstacles. Experimental results demonstrate the effectiveness of the proposed method. ETPNav yields more than **10%** and **20%** improvements over prior state-of-the-art on R2R-CE and RxR-CE datasets, respectively.

<div align="center">
    <img src="assets/overview.png", width="1000">
    <img src="assets/mapping.png", width="1000">
</div>

Leadboard:

<div align="center">
    <img src="assets/sota.png", width="1000">
</div>

## TODO's

* [ ] Tidy and release the R2R-CE fine-tuning code.
* [ ] Tidy and release the RxR-CE fine-tuning code.
* [ ] Release the pre-training code.
* [ ] Release the checkpoints.

## Setup

### Installation

Follow the [Habitat Installation Guide](https://github.com/facebookresearch/habitat-lab#installation) to install [`habitat-lab`](https://github.com/facebookresearch/habitat-lab) and [`habitat-sim`](https://github.com/facebookresearch/habitat-sim). We use version [`v0.1.7`](https://github.com/facebookresearch/habitat-lab/releases/tag/v0.1.7) in our experiments, same as in the VLN-CE, please refer to the [VLN-CE](https://github.com/jacobkrantz/VLN-CE) page for more details. In brief:

1. Create a virtual environment. We develop this project with Python 3.6.

   ```bash
   conda create -n etpnav python=3.6
   conda activate etpnav
   ```
2. Install `habitat-sim` for a machine with multiple GPUs or without an attached display (i.e. a cluster):

   ```bash
   conda install -c aihabitat -c conda-forge habitat-sim=0.1.7 headless
   ```
3. Clone this repository and install all requirements for `habitat-lab`, VLN-CE and our experiments. Note that we specify `gym==0.21.0` because its latest version is not compatible with `habitat-lab-v0.1.7`.

   ```bash
   git clone git@github.com:MarSaKi/ETPNav.git
   cd ETPNav
   python -m pip install -r requirements.txt
   pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
   ```
4. Clone a stable `habitat-lab` version from the github repository and install. The command below will install the core of Habitat Lab as well as the habitat_baselines.

   ```bash
   git clone --branch v0.1.7 git@github.com:facebookresearch/habitat-lab.git
   cd habitat-lab
   python setup.py develop --all # install habitat and habitat_baselines
   ```

### Scenes: Matterport3D

Instructions copied from [VLN-CE](https://github.com/jacobkrantz/VLN-CE):

Matterport3D (MP3D) scene reconstructions are used. The official Matterport3D download script (`download_mp.py`) can be accessed by following the instructions on their [project webpage](https://niessner.github.io/Matterport/). The scene data can then be downloaded:

```bash
# requires running with python 2.7
python download_mp.py --task habitat -o data/scene_datasets/mp3d/
```

Extract such that it has the form `scene_datasets/mp3d/{scene}/{scene}.glb`. There should be 90 scenes. Place the `scene_datasets` folder in `data/`.

## Running

Training and Evaluation

Use `release_r2r.bash` and `release_rxr.bash` for `Training/Evaluation/Inference with a single GPU or with multiple GPUs on a single node. `Simply adjust the arguments of the bash scripts:

```
# for R2R-CE
CUDA_VISIBLE_DEVICES=0,1 bash run/release_r2r.bash train 12345  # training
CUDA_VISIBLE_DEVICES=0,1 bash run/release_r2r.bash eval 12345   # evaluation
CUDA_VISIBLE_DEVICES=0,1 bash run/release_r2r.bash inter 12345  # inference
```

```
# for RxR-CE
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run/release_rxr.bash train 12345  # training
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run/release_rxr.bash eval 12345   # evaluation
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run/release_rxr.bash inter 12345  # inference
```

# Contact Information

* dong.an@cripac.ia.ac.cn, [Dong An](https://marsaki.github.io/)
* hanqingwang@bit.edu.cn, [Hanqing Wang](https://hanqingwangai.github.io/)
* wenguanwang.ai@gmail.com, [Wenguan Wang](https://sites.google.com/view/wenguanwang)
* yhuang@nlpr.ia.ac.cn, [Yan Huang](https://yanrockhuang.github.io/)

# Acknowledge

Our implementations are partially inspired by [CWP](https://github.com/YicongHong/Discrete-Continuous-VLN), [Sim2Sim ](https://github.com/jacobkrantz/Sim2Sim-VLNCE)and [DUET](https://github.com/cshizhe/VLN-DUET).

Thanks for their great works!

# Citation

If you find this repository is useful, please consider cite our work:

```
@article{an2023etpnav,
  title={ETPNav: Evolving Topological Planning for Vision-Language Navigation in Continuous Environments}, 
  author={An, Dong and Wang, Hanqing and Wang, Wenguan and Wang, Zun and Huang, Yan and He, Keji and Wang, Liang},
  journal={arXiv preprint arXiv:2304.03047}
  year={2023},
}
```
