#!/bin/bash
#SBATCH --job-name=eval_qm9_mdlm_proseco
#SBATCH --output=slurm_output/eval_qm9_mdlm_proseco.out
#SBATCH --error=slurm_output/eval_qm9_mdlm_proseco.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=spgpu2
#SBATCH --account=jjparkcv_owned1
#SBATCH --time=8:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4

# Submit with: sbatch eval_qm9_mdlm_proseco.sh
# Optional overrides:
#   PROP=<qed|ring_count>     (default: qed)
#   SAMPLING_STEPS=<int>      (default: 32)   total NFE budget
#   PROSECO_BUDGET=<int>      (default: 3)    S: corrector passes
#   PROSECO_FREQ=<int>        (default: 1)    omega: corrector frequency
#   SEED=<int>                (default: 1)

export HF_HOME=/nfs/turbo/coe-jjparkcv-medium/satyam/.cache/huggingface
export HYDRA_FULL_ERROR=1
export WANDB_DISABLED=true

OUTPUT_DIR_BASE=/nfs/turbo/coe-jjparkcv-medium/satyam/molecule_gen/
RUN_NAME=mdlm_no-guidance_proseco
DATA_CACHE_DIR=/nfs/turbo/coe-jjparkcv-medium/satyam/.cache
CKPT="${OUTPUT_DIR_BASE}/${RUN_NAME}"
RES="./eval_results"

PROP="${PROP:-qed}"
SAMPLING_STEPS="${SAMPLING_STEPS:-32}"
PROSECO_BUDGET="${PROSECO_BUDGET:-1}"
PROSECO_FREQ="${PROSECO_FREQ:-1}"
SEED="${SEED:-1}"

results_csv_path="${RES}/qm9-eval-uncond_proseco_${PROP}_T-${SAMPLING_STEPS}_S-${PROSECO_BUDGET}_w-${PROSECO_FREQ}_seed-${SEED}.csv"
generated_seqs_path="${RES}/samples-qm9-eval-uncond_proseco_${PROP}_T-${SAMPLING_STEPS}_S-${PROSECO_BUDGET}_w-${PROSECO_FREQ}_seed-${SEED}.json"

PYTHONPATH=. python -u guidance_eval/qm9_eval.py \
    hydra.output_subdir=null \
    hydra.run.dir="${CKPT}" \
    hydra/job_logging=disabled \
    hydra/hydra_logging=disabled \
    seed=${SEED} \
    mode=qm9_eval \
    eval.checkpoint_path="${CKPT}/checkpoints/best.ckpt" \
    data=qm9 \
    data.cache_dir="${DATA_CACHE_DIR}" \
    data.label_col="${PROP}" \
    data.label_col_pctile=90 \
    data.num_classes=2 \
    model=small \
    backbone=dit \
    model.length=32 \
    parameterization=subs \
    diffusion=absorbing_state \
    time_conditioning=False \
    T=0 \
    training.guidance=null \
    training.refine=True \
    training.proseco=True \
    training.refine_remask_ratio=t \
    training.refine_temperature=0.0 \
    training.refine_loss_type=sum \
    training.compute_loss_on_pad_tokens=True \
    training.use_simple_ce_loss=False \
    zero_recon_loss=False \
    sampling.num_sample_batches=64 \
    sampling.batch_size=16 \
    sampling.steps=${SAMPLING_STEPS} \
    sampling.use_cache=False \
    sampling.method=proseco \
    sampling.proseco_budget=${PROSECO_BUDGET} \
    sampling.proseco_freq=${PROSECO_FREQ} \
    sampling.corrector_sampling=argmax \
    +eval.results_csv_path=${results_csv_path} \
    eval.generated_samples_path=${generated_seqs_path}
