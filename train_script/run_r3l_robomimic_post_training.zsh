#!/usr/bin/env zsh
set -euo pipefail

# Run R3L-style residual post-training on Robomimic image tasks.
#
# Usage:
#   zsh train_script/run_r3l_robomimic_post_training.zsh
#   zsh train_script/run_r3l_robomimic_post_training.zsh square device=cuda:1 seed=0
#   DRY_RUN=1 zsh train_script/run_r3l_robomimic_post_training.zsh transport

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${DICE_RL_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${ROOT_DIR}"

DICE_RL_ASSET_ROOT="${DICE_RL_ASSET_ROOT:-/home/wenkai001/ssd/ziming/dice-rl}"
export DICE_RL_DATA_DIR="${DICE_RL_DATA_DIR:-${DICE_RL_ASSET_ROOT}/data_dir}"
export DICE_RL_LOG_DIR="${DICE_RL_LOG_DIR:-${ROOT_DIR}/log_dir}"
DICE_RL_CKPT_LOG_DIR="${DICE_RL_CKPT_LOG_DIR:-${DICE_RL_ASSET_ROOT}/log_dir}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${DRY_RUN:-0}" != "1" ]] && ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  print -u2 "Python executable not found: ${PYTHON_BIN}"
  print -u2 "Run 'conda activate dice-rl' first, or set PYTHON_BIN=/path/to/python."
  exit 127
fi

typeset -a default_tasks selected_tasks hydra_overrides
default_tasks=(square tool_hang transport)
selected_tasks=()
hydra_overrides=()

while (($#)); do
  case "$1" in
    all)
      selected_tasks=("${default_tasks[@]}")
      ;;
    square|tool_hang|transport)
      selected_tasks+=("$1")
      ;;
    *)
      hydra_overrides+=("$1")
      ;;
  esac
  shift
done

if ((${#selected_tasks[@]} == 0)); then
  selected_tasks=("${default_tasks[@]}")
fi

if [[ "${R3L_REMAP_CUDA_VISIBLE_DEVICES:-1}" == "1" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  for i in {1..${#hydra_overrides[@]}}; do
    if [[ "${hydra_overrides[$i]}" == device=cuda:<-> ]]; then
      gpu_id="${hydra_overrides[$i]#device=cuda:}"
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      export EGL_DEVICE_ID=0
      export MUJOCO_EGL_DEVICE_ID=0
      hydra_overrides[$i]="device=cuda:0"
      print "[R3L] Remapped physical GPU ${gpu_id} via CUDA_VISIBLE_DEVICES; Hydra device=cuda:0; MUJOCO_EGL_DEVICE_ID=0"
      break
    fi
  done
fi

typeset -A ckpt_path data_dir_name
ckpt_path=(
  square     "${DICE_RL_CKPT_LOG_DIR}/robomimic-pretrain/pretrained_bc_policy_square_img/checkpoint/state_2000.pt"
  tool_hang  "${DICE_RL_CKPT_LOG_DIR}/robomimic-pretrain/tool_hang_img/checkpoint/state_1400.pt"
  transport  "${DICE_RL_CKPT_LOG_DIR}/robomimic-pretrain/transport_img/checkpoint/state_2400.pt"
)
data_dir_name=(
  square     "square-img"
  tool_hang  "tool-hang-img"
  transport  "transport-img"
)

max_correction="${R3L_MAX_CORRECTION:-0.15}"
# Calibrated for the bounded residual: with tanh/softsign * 0.15 the worst-case
# ||residual||^2 ~= max_correction^2 = 0.0225, so a coefficient ~ O(20) gives a
# saturated-residual penalty of ~0.45, comparable in scale to the normalized
# Q-loss (~O(1)) and prevents the actor from drifting into saturation.
l2_penalty_coeff="${R3L_L2_PENALTY_COEFF:-20.0}"
q_chunk_num_samples="${R3L_Q_CHUNK_NUM_SAMPLES:-16}"
q_chunk_warmup_steps="${R3L_Q_CHUNK_WARMUP_STEPS:-10000}"
residual_squash="${R3L_RESIDUAL_SQUASH:-softsign}"
zero_init_final="${R3L_ZERO_INIT_FINAL:-true}"
max_correction_init="${R3L_MAX_CORRECTION_INIT:-0.03}"
max_correction_warmup_steps="${R3L_MAX_CORRECTION_WARMUP:-30000}"
critic_only_warmup_steps="${R3L_CRITIC_ONLY_WARMUP:-5000}"
use_soft_q_filtering="${R3L_USE_SOFT_Q_FILTERING:-true}"
# Threshold for q_overestimation < threshold to count as underestimated.
# A large positive value makes the underestimation gate always-on, so soft
# Q-filtering reduces to "drop BC penalty whenever the residual policy beats
# the pretrained policy in Q" (i.e. self-imitation on better samples).
q_underestimation_threshold="${R3L_Q_UNDERESTIMATION_THRESHOLD:-1000000.0}"

for task in "${selected_tasks[@]}"; do
  config_dir="${ROOT_DIR}/cfg/robomimic/finetune/${task}"
  config_name="ft_distill_residual_flow_unet_img"
  task_data_dir="${DICE_RL_DATA_DIR}/robomimic/${data_dir_name[$task]}"
  normalization_path="${task_data_dir}/ph_pretrain/normalization.npz"
  dataset_path="${task_data_dir}/ph_finetune/train.npz"

  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    if [[ ! -f "${config_dir}/${config_name}.yaml" ]]; then
      print -u2 "Missing config: ${config_dir}/${config_name}.yaml"
      exit 1
    fi
    if [[ ! -f "${ckpt_path[$task]}" ]]; then
      print -u2 "Missing pretrained checkpoint for ${task}: ${ckpt_path[$task]}"
      exit 1
    fi
    if [[ ! -f "${normalization_path}" ]]; then
      print -u2 "Missing normalization file for ${task}: ${normalization_path}"
      exit 1
    fi
    if [[ ! -f "${dataset_path}" ]]; then
      print -u2 "Missing finetune dataset for ${task}: ${dataset_path}"
      exit 1
    fi
  fi

  cmd=(
    "${PYTHON_BIN}" script/run.py
    --config-dir="${config_dir}"
    --config-name="${config_name}"
    "base_policy_path=${ckpt_path[$task]}"
    "normalization_path=${normalization_path}"
    "expert_dataset.dataset_path=${dataset_path}"
    "wandb.project=robomimic-${task}-r3l-post-training-img"
    "name=${task}_r3l_residual_flow_unet_img"
    "_target_=agent.finetune.train_distill_residual_flow_img_agent.TrainDistillResidualFlowImgAgent"
    "model._target_=model.rl.r3l_residual_rl_img.R3LResidualRLImgModel"
    "++model.max_correction=${max_correction}"
    "++model.clip_final_action=true"
    "++model.bc_loss_weight=${l2_penalty_coeff}"
    "++model.q_chunk_num_samples=${q_chunk_num_samples}"
    "++model.q_chunk_critic_reduction=min"
    "++model.q_chunk_warmup_steps=${q_chunk_warmup_steps}"
    "++model.residual_squash=${residual_squash}"
    "++model.zero_init_final=${zero_init_final}"
    "++model.max_correction_init=${max_correction_init}"
    "++model.max_correction_warmup_steps=${max_correction_warmup_steps}"
    "++model.use_soft_q_filtering=${use_soft_q_filtering}"
    "++model.q_underestimation_threshold=${q_underestimation_threshold}"
    "++train.critic_only_warmup_steps=${critic_only_warmup_steps}"
    "++online_explore_strategy=r3l_q_chunk"
    "++evaluate_strategy=r3l_q_chunk"
    "++num_exploration_samples=${q_chunk_num_samples}"
    "${hydra_overrides[@]}"
  )

  print "\n[R3L] ${task}"
  print -r -- "${cmd[@]}"
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    "${cmd[@]}"
  fi
done
