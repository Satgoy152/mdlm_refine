#!/bin/bash
#SBATCH --job-name=eval_recon
#SBATCH --output=slurm_output/eval_recon_%j.out
#SBATCH --error=slurm_output/eval_recon_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=spgpu2
#SBATCH --account=jjparkcv_owned1
#SBATCH --time=4:00:00
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=16


python eval_reconstruct.py \
  --model refine \
  --sampler refine  \
  --input_file sample_wikitext.jsonl \
  --n_samples 100 \
  --steps 128 \
  --mask_frac 0.25 \
  --corrupt_frac 0.25 \
  --eval_batch_size 10 \
  --n_correct_per_step 10 \
  --proseco_budget 1 \
  --proseco_freq 1 \
  --corrector_sampling argmax \
  --save_gen_dir ./eval_reconstruct_samples \
  --save_res_dir ./eval_results
