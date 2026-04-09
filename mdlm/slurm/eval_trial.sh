#!/bin/bash
#SBATCH --job-name=eval_trial
#SBATCH --output=eval_trial.out
#SBATCH --error=eval_trial.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=spgpu2
#SBATCH --account=jjparkcv_owned1
#SBATCH --time=4:00:00
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=16

# source /home/sagoyal/research/mdlm/.venv/bin/activate

# cd /home/sagoyal/research/mdlm

python model_eval.py \
  --models refine \
  --samplers refine \
  --num_samples 100 \
  --num_steps 128 \
  --gen_len 1024 \
  --metrics gen_ppl rep_n \
  --gen_model gpt2 \
  --save_gen_dir ./eval_reconstruct_samples \
  --save_res_dir ./eval_results \
  --corrector_sampling 'topk' \
  --eval_batch_size 10