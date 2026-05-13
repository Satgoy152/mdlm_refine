#!/bin/bash
#SBATCH --job-name=eval_qm9_mdlm_refine
#SBATCH --output=slurm_output/eval_qm9_mdlm_refine.out
#SBATCH --error=slurm_output/eval_qm9_mdlm_refine.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=spgpu2
#SBATCH --account=jjparkcv_owned1
#SBATCH --time=8:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4

# Submit with: sbatch eval_qm9_mdlm_refine.sh
# Optional overrides:
#   PROP=<qed|ring_count>  (default: qed) - property to compute stats for
#   SAMPLING_STEPS=<int>   (default: 32)
#   SEED=<int>             (default: 1)

export HF_HOME=/nfs/turbo/coe-jjparkcv-medium/satyam/.cache/huggingface
export HYDRA_FULL_ERROR=1
export WANDB_DISABLED=true

# source /home/sagoyal/research/mdlm_refine/discrete-diffusion-guidance/.venv/bin/activate
# cd /home/sagoyal/research/mdlm_refine/discrete-diffusion-guidance || exit 1
MODEL=refine
OUTPUT_DIR_BASE=/nfs/turbo/coe-jjparkcv-medium/satyam/molecule_gen/
RUN_NAME=mdlm_no-guidance_${MODEL}
DATA_CACHE_DIR=/nfs/turbo/coe-jjparkcv-medium/satyam/.cache
CKPT="${OUTPUT_DIR_BASE}/${RUN_NAME}"
RES="./eval_results"

PROP="${PROP:-qed}"
SAMPLING_STEPS="${SAMPLING_STEPS:-32}"
SEED="${SEED:-1}"

# Unconditional sampling via CFG with gamma=0.0
# (collapses to unconditional logits in diffusion.py)
results_csv_path="${RES}/qm9-eval-uncond_${PROP}_T-${SAMPLING_STEPS}_seed-${SEED}.csv"
generated_seqs_path="${RES}/samples-qm9-eval-uncond_${PROP}_T-${SAMPLING_STEPS}_seed-${SEED}.json"

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
    training.proseco=False \
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
    sampling.method=vanilla \
    sampling.n_correct_per_step=32 \
    sampling.correct_mode=topk \
    sampling.correct_threshold=0.01 \
    +eval.results_csv_path=${results_csv_path} \
    eval.generated_samples_path=${generated_seqs_path}
