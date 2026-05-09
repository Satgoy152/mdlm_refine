#!/bin/bash
#SBATCH -o ../watch_folder/%x_%j.out  # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --constraint="[a100|a6000|a5000|3090]"
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

<<comment
#  Usage:
cd scripts/
MODEL=<ar|mdlm|udlm>
sbatch \
  --export=ALL,MODEL=${MODEL} \
  --job-name=train_qm9_no-guidance_${MODEL} \
  train_qm9_no-guidance.sh
comment

# Setup environment
source /home/sagoyal/research/mdlm_refine/discrete-diffusion-guidance/.venv/bin/activate
cd /home/sagoyal/research/mdlm_refine/discrete-diffusion-guidance || exit 1
export NCCL_P2P_LEVEL=NVL
export HYDRA_FULL_ERROR=1

# Expecting:
#  - MODEL (ar, mdlm, udlm)
#  - USE_SIMPLE_CE_LOSS (True, False; optional, default: False)
#  - REFINE / PROSECO (True, False; optional, MDLM only)
#  - REFINE_REMASK_RATIO ('random' | 't' | float; optional, default: t)
#  - REFINE_TEMPERATURE (float; optional, default: 0.0)
#  - REFINE_LOSS_TYPE ('sum' | 'mean' | 'pass2_only'; optional, default: sum)
#  - OUTPUT_DIR_BASE (path; optional, default: ${PWD}/outputs/qm9)
if [ -z "${MODEL}" ]; then
  echo "MODEL is not set"
  exit 1
fi
if [ -z "${USE_SIMPLE_CE_LOSS}" ]; then
  USE_SIMPLE_CE_LOSS=False
fi
REFINE="${REFINE:-False}"
PROSECO="${PROSECO:-False}"
REFINE_REMASK_RATIO="${REFINE_REMASK_RATIO:-t}"
REFINE_TEMPERATURE="${REFINE_TEMPERATURE:-0.0}"
REFINE_LOSS_TYPE="${REFINE_LOSS_TYPE:-sum}"

RUN_NAME="${MODEL}_no-guidance"
if [ "${USE_SIMPLE_CE_LOSS}" = "True" ]; then
  RUN_NAME="${RUN_NAME}_simple-ce"
fi
if [ "${PROSECO}" = "True" ]; then
  RUN_NAME="${RUN_NAME}_proseco"
elif [ "${REFINE}" = "True" ]; then
  RUN_NAME="${RUN_NAME}_refine"
fi

OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-${PWD}/outputs/qm9}"
WANDB_NAME="${WANDB_NAME:-qm9_${RUN_NAME}}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-/nfs/turbo/coe-jjparkcv-medium/satyam/.cache}"

EXTRA_FLAGS=()
if [ "${REFINE}" = "True" ]; then
  EXTRA_FLAGS+=( "training.refine=True" )
fi
if [ "${PROSECO}" = "True" ]; then
  EXTRA_FLAGS+=( "training.proseco=True" )
fi
if [ "${REFINE}" = "True" ] || [ "${PROSECO}" = "True" ]; then
  EXTRA_FLAGS+=(
    "training.refine_remask_ratio=${REFINE_REMASK_RATIO}"
    "training.refine_temperature=${REFINE_TEMPERATURE}"
    "training.refine_loss_type=${REFINE_LOSS_TYPE}"
  )
fi

if [ "${MODEL}" = "ar" ]; then
  # AR
  DIFFUSION="absorbing_state"
  PARAMETERIZATION="ar"
  T=0
  TIME_COND=False
  ZERO_RECON_LOSS=False
elif [ "${MODEL}" = "mdlm" ]; then
  # MDLM
  DIFFUSION="absorbing_state"
  PARAMETERIZATION="subs"
  T=0
  TIME_COND=False
  ZERO_RECON_LOSS=False
elif [ "${MODEL}" = "udlm" ]; then
  # UDLM
  DIFFUSION="uniform"
  PARAMETERIZATION="d3pm"
  T=0
  TIME_COND=True
  ZERO_RECON_LOSS=True
else
  echo "MODEL must be one of ar, mdlm, udlm"
  exit 1
fi

# To enable preemption re-loading, set `hydra.run.dir` or
# `checkpointing.save_dir` explicitly.
srun python -u -m main \
  diffusion="${DIFFUSION}" \
  parameterization="${PARAMETERIZATION}" \
  T=${T} \
  time_conditioning=${TIME_COND} \
  zero_recon_loss=${ZERO_RECON_LOSS} \
  data=qm9 \
  data.cache_dir="${DATA_CACHE_DIR}" \
  data.label_col=null \
  data.label_col_pctile=null \
  data.num_classes=null \
  eval.generate_samples=False \
  loader.global_batch_size=2048 \
  loader.eval_global_batch_size=4096 \
  backbone="dit" \
  model=small \
  model.length=32 \
  optim.lr=3e-4 \
  lr_scheduler=cosine_decay_warmup \
  lr_scheduler.warmup_t=1000 \
  lr_scheduler.lr_min=3e-6 \
  training.guidance=null \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5_000 \
  training.compute_loss_on_pad_tokens=True \
  training.use_simple_ce_loss=${USE_SIMPLE_CE_LOSS} \
  trainer.max_steps=25_000 \
  trainer.val_check_interval=1.0 \
  wandb.name="${WANDB_NAME}" \
  hydra.run.dir="${OUTPUT_DIR_BASE}/${RUN_NAME}" \
  "${EXTRA_FLAGS[@]}"
