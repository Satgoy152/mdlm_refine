#!/bin/bash
#SBATCH --job-name=eval_confidence
#SBATCH --output=eval_confidence.out
#SBATCH --error=eval_confidence.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition=spgpu2
#SBATCH --account=jjparkcv_owned1
#SBATCH --time=24:00:00
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=16

source /home/sagoyal/research/mdlm/.venv/bin/activate

# ── configuration ────────────────────────────────────────────────────────────
CHECKPOINT="/home/sagoyal/research/mdlm/outputs/openwebtext-train/2026.02.25/184842/checkpoints/best.ckpt"
DATA_CACHE_DIR="/nfs/turbo/coe-jjparkcv-medium/satyam/.cache/datasets"
OUTPUT_FILE="samples_confidence_t128.jsonl"
# ─────────────────────────────────────────────────────────────────────────────

# cd /home/sagoyal/research/mdlm

python eval_sample.py \
  --checkpoint "${CHECKPOINT}" \
  --n_samples 512 \
  --steps 128 \
  --method confidence \
  --temperature 1.0 \
  --save_samples \
  --output_file "${OUTPUT_FILE}" \
  --eval_nll \
  -data openwebtext-split \
  --data_cache_dir "${DATA_CACHE_DIR}" \
  --seed 42

echo "Sampling done. Computing generative PPL..."

python eval_ppl.py \
  --samples_file "${OUTPUT_FILE}" \
  --model gpt2
