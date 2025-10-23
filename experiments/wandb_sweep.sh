#!/bin/bash
#SBATCH --job-name=wandb_sweep
#SBATCH --output=.logs/wandb_sweep_%j_%a.out
#SBATCH --error=.logs/wandb_sweep_%j_%a.err
#SBATCH --gres=gpu:1
#SBATCH -C GPU_SKU:H100_SXM5
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL

usage() {
  echo "Usage: $0 SWEEP_ID VENV_PATH WANDB_USER"
  echo "  SWEEP_ID       e.g. y4g2tzg7"
  echo "  VENV_PATH      path to your virtualenv (dir that contains bin/activate)"
  echo "  WANDB_USER     your wandb username"
  exit 1
}

# require 3 args
[[ $# -lt 3 ]] && usage

SWEEP_ID=$1
VENV_PATH=$2
WANDB_USER=$3
WANDB_PROJECT="predictability"

[[ -z "$SWEEP_ID" || -z "$VENV_PATH" || -z "$WANDB_USER" ]] && usage

mkdir -p .logs

# Activate the virtual environment
ACTIVATE="$VENV_PATH/bin/activate"
if [[ ! -f "$ACTIVATE" ]]; then
  echo "Error: could not find $ACTIVATE"
  exit 2
fi
source "$ACTIVATE"

echo "Running: wandb agent ${WANDB_USER}/${WANDB_PROJECT}/${SWEEP_ID}"
python -m wandb agent "${WANDB_USER}/${WANDB_PROJECT}/${SWEEP_ID}" || {
  echo "wandb agent exited with status code $?. This is expected if the sweep is completed."
  exit 0
}
