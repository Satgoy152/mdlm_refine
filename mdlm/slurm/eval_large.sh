#!/bin/bash
#SBATCH --job-name=eval_large
#SBATCH --output=slurm_output/eval_large.out
#SBATCH --error=slurm_output/eval_large.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=spgpu2
#SBATCH --account=jjparkcv_owned1
#SBATCH --time=48:00:00
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=16

source /home/sagoyal/research/mdlm/.venv/bin/activate

# cd /home/sagoyal/research/mdlm

python model_eval.py \
  --models vanilla \
  --samplers refine \
  --num_samples 100 \
  --num_steps 128 256 512 1024 \
  --gen_len 1024 \
  --metrics gen_ppl \
  --gen_model gpt2 \
  --save_gen_dir ./eval_samples \
  --save_res_dir ./eval_results \
  --eval_batch_size 10 \
  --corrector_sampling 'topk' \
