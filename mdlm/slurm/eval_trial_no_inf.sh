#!/bin/bash
#SBATCH --job-name=all_results
#SBATCH --output=slurm_output/all_results.out
#SBATCH --error=slurm_output/all_results.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=spgpu2
#SBATCH --account=jjparkcv_owned1
#SBATCH --time=24:00:00
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=16

# source /home/sagoyal/research/mdlm/.eval-venv/bin/activate

# cd /home/sagoyal/research/mdlm

python model_eval.py \
  --models refine\
  --samplers refine\
  --num_samples 100 \
  --num_steps 128 256 512 1024 \
  --gen_len 1024 \
  --metrics gen_ppl entropy \
  --gen_model llama1b \
  --save_gen_dir ./eval_samples \
  --save_res_dir ./eval_results \
  --eval_batch_size 16 \
  --skip_inference \
