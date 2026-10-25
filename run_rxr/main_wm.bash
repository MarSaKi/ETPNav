# ETPNav + bidirectional world models + corrective memory (trainer SS-ETP-WM).
# Usage: bash run_rxr/main_wm.bash {train|eval|infer} <master_port> [variant]
#   variant: full (default) | fwd | bidir | bidir_mem | mem_only   (see docs/WM_DESIGN.md)
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

variant=${3:-full}
case $variant in
      full)      ablation="" ;;
      fwd)       ablation="WM.use_inverse False WM.use_exec False WM.use_memory False WM.use_correction False WM.stop_veto False" ;;
      bidir)     ablation="WM.use_memory False WM.use_correction False WM.stop_veto False" ;;
      bidir_mem) ablation="WM.use_correction False WM.stop_veto False" ;;
      mem_only)  ablation="WM.use_forward False WM.use_inverse False WM.use_exec False" ;;
      *) echo "unknown variant $variant"; exit 1 ;;
esac
exp=wm_rxr_${variant}
backbone=data/logs/checkpoints/release_rxr/ckpt.iter19600.pth   # fine-tuned ETPNav (A0)

common="--exp-config run_rxr/iter_train.yaml
      TRAINER_NAME SS-ETP-WM
      SIMULATOR_GPU_IDS [0,1]
      TORCH_GPU_IDS [0,1]
      GPU_NUMBERS 2
      NUM_ENVIRONMENTS 8
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      WM.backbone_ckpt $backbone
      IL.expert_policy ndtw
      $ablation"

flag1="--exp_name $exp --run-type train $common
      IL.iters 15000
      IL.log_every 500
      IL.ml_weight 1.0
      IL.sample_ratio 0.75
      IL.decay_interval 2000
      IL.load_from_ckpt True
      IL.is_requeue True
      IL.waypoint_aug True
      WM.lr 1e-4
      WM.perturb_prob 0.1
      "

flag2="--exp_name $exp --run-type eval $common
      EVAL.CKPT_PATH_DIR data/logs/checkpoints/$exp
      IL.back_algo control
      "

flag3="--exp_name $exp --run-type inference $common
      INFERENCE.CKPT_PATH data/logs/checkpoints/$exp/ckpt.iter15000.pth
      INFERENCE.PREDICTIONS_FILE preds_$exp.json
      IL.back_algo control
      "

mode=$1
case $mode in
      train)
      echo "###### train mode ($variant) ######"
      python -m torch.distributed.launch --nproc_per_node=2 --master_port $2 run.py $flag1
      ;;
      eval)
      echo "###### eval mode ($variant) ######"
      python -m torch.distributed.launch --nproc_per_node=2 --master_port $2 run.py $flag2
      ;;
      infer)
      echo "###### infer mode ($variant) ######"
      python -m torch.distributed.launch --nproc_per_node=2 --master_port $2 run.py $flag3
      ;;
esac
