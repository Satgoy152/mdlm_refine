#!/bin/bash
#SBATCH --job-name=eval_proseco
#SBATCH --output=eval_proseco.out
#SBATCH --error=eval_prseco.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=spgpu2
#SBATCH --account=jjparkcv_owned1
#SBATCH --time=24:00:00
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=16

# source /home/sagoyal/research/mdlm/.venv/bin/activate

# cd /home/sagoyal/research/mdlm

python model_eval.py \
  --models proseco \
  --samplers proseco refine_ddpm_onepass \
  --num_samples 1000 \
  --num_steps 128 256 512 1024 \
  --gen_len 1024 \
  --metrics gen_ppl \
  --gen_model gpt2 \
  --save_gen_dir ./eval_samples \
  --save_res_dir ./eval_results \
  --eval_batch_size 16
