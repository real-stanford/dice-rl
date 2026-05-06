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

export DICE_RL_DATA_DIR="${DICE_RL_DATA_DIR:-${ROOT_DIR}/data_dir}"
export DICE_RL_LOG_DIR="${DICE_RL_LOG_DIR:-${ROOT_DIR}/log_dir}"
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

typeset -A ckpt_path data_dir_name
ckpt_path=(
  square     "${ROOT_DIR}/log_dir/robomimic-pretrain/pretrained_bc_policy_square_img/checkpoint/state_2000.pt"
  tool_hang  "${ROOT_DIR}/log_dir/robomimic-pretrain/tool_hang_img/checkpoint/state_1400.pt"
  transport  "${ROOT_DIR}/log_dir/robomimic-pretrain/transport_img/checkpoint/state_2400.pt"
)
data_dir_name=(
  square     "square-img"
  tool_hang  "tool-hang-img"
  transport  "transport-img"
)

base_discount="${R3L_BASE_DISCOUNT:-0.999}"
query_freq="${R3L_QUERY_FREQ:-4}"
max_correction="${R3L_MAX_CORRECTION:-0.15}"
l2_penalty_coeff="${R3L_L2_PENALTY_COEFF:-0.3}"
q_chunk_num_samples="${R3L_Q_CHUNK_NUM_SAMPLES:-4}"
q_chunk_warmup_steps="${R3L_Q_CHUNK_WARMUP_STEPS:-50000}"

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
    "model.max_correction=${max_correction}"
    "model.bc_loss_weight=${l2_penalty_coeff}"
    "model.r3l_query_discount_steps=${query_freq}"
    "model.q_chunk_num_samples=${q_chunk_num_samples}"
    "model.q_chunk_critic_reduction=min"
    "model.q_chunk_warmup_steps=${q_chunk_warmup_steps}"
    "model.multi_sample_next_noise=true"
    "model.num_next_noise_samples=${q_chunk_num_samples}"
    "model.critic_ensemble_size=10"
    "model.conservative_q_method=min"
    "train.gamma=${base_discount}"
    "online_explore_strategy=r3l_q_chunk"
    "evaluate_strategy=r3l_q_chunk"
    "num_exploration_samples=${q_chunk_num_samples}"
    "use_rlpd=false"
    "use_adaptive_expert_ratio=false"
    "replay_buffer.use_n_step=false"
    "replay_buffer.expert_use_n_step=false"
    "expert_dataset.use_n_step=false"
    "expert_dataset.n_step=1"
    "${hydra_overrides[@]}"
  )

  print "\n[R3L] ${task}"
  print -r -- "${cmd[@]}"
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    "${cmd[@]}"
  fi
done
