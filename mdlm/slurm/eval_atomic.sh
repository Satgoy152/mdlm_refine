#!/bin/bash
#SBATCH --job-name=eval_reconstruct
#SBATCH --output=eval_reconstruct.out
#SBATCH --error=eval_reconstruct.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=spgpu2
#SBATCH --account=jjparkcv_owned1
#SBATCH --time=2:00:00
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=8

# source /home/sagoyal/research/mdlm/.venv/bin/activate

# ── configuration ────────────────────────────────────────────────────────────
INPUT_FILE="vis_samples.jsonl"
# ─────────────────────────────────────────────────────────────────────────────

python eval_reconstruct.py \
  --model refine \
  --sampler refine \
  --input_file "${INPUT_FILE}" \
  --n_samples 1 \
  --atomic_scenario B \
  --mask_frac 0.25 \
  --corrupt_frac 0.25 \
  --steps 1 \
  --verbose \
  --seed 44 \
  --n_correct_per_step 20 \
  # --show_intermediate \
  