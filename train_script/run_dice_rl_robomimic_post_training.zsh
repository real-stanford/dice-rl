#!/usr/bin/env zsh
set -euo pipefail

# Run DICE-RL post-training on Robomimic image tasks.
# Usage:
#   zsh train_script/run_dice_rl_robomimic_post_training.zsh
#   zsh train_script/run_dice_rl_robomimic_post_training.zsh square device=cuda:1 seed=0
#   DRY_RUN=1 zsh train_script/run_dice_rl_robomimic_post_training.zsh transport

ROOT_DIR="${DICE_RL_ROOT:-/home/wenkai001/ssd/ziming/dice-rl}"
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

for task in "${selected_tasks[@]}"; do
  config_dir="${ROOT_DIR}/cfg/robomimic/finetune/${task}"
  config_name="ft_distill_residual_flow_unet_img"
  task_data_dir="${DICE_RL_DATA_DIR}/robomimic/${data_dir_name[$task]}"
  normalization_path="${task_data_dir}/ph_pretrain/normalization.npz"
  dataset_path="${task_data_dir}/ph_finetune/train.npz"

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

  cmd=(
    "${PYTHON_BIN}" script/run.py
    --config-dir="${config_dir}"
    --config-name="${config_name}"
    "base_policy_path=${ckpt_path[$task]}"
    "normalization_path=${normalization_path}"
    "expert_dataset.dataset_path=${dataset_path}"
    "wandb.project=robomimic-${task}-dice-rl-post-training-img"
    "${hydra_overrides[@]}"
  )

  print "\n[DICE-RL] ${task}"
  print -r -- "${cmd[@]}"
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    "${cmd[@]}"
  fi
done
