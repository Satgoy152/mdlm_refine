#!/bin/bash
#SBATCH --job-name=eval_mauve
#SBATCH --output=slurm_output/eval_mauve.out
#SBATCH --error=slurm_output/eval_mauve.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=spgpu2
#SBATCH --account=jjparkcv_owned1
#SBATCH --time=24:00:00
#SBATCH --mem=10GB
#SBATCH --cpus-per-task=16

source /home/sagoyal/research/mdlm/.eval-venv/bin/activate

# cd /home/sagoyal/research/mdlm

python model_eval.py \
  --models proseco \
  --samplers proseco \
  --num_samples 100 \
  --num_steps 128 \
  --gen_len 1024 \
  --metrics mauve \
  --gen_model llama1b \
  --mauve_dataset openwebtext-split \
  --cache_dir /nfs/turbo/coe-jjparkcv-medium/satyam/.cache/datasets \
  --save_gen_dir ./eval_reconstruct_samples \
  --save_res_dir ./eval_results \
  --eval_batch_size 16 \
  --skip_inference \
