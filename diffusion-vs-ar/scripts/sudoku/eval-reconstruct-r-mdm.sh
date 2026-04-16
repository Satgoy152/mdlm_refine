#!/bin/bash
#SBATCH --job-name=eval_recon_sudoku
#SBATCH --output=slurm_output/eval_recon_sudoku_%j.out
#SBATCH --error=slurm_output/eval_recon_sudoku_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=spgpu2
#SBATCH --account=jjparkcv_owned1
#SBATCH --time=4:00:00
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=8

export WANDB_DISABLED=true

exp=/nfs/turbo/coe-jjparkcv-medium/satyam/sr/sudoku/refine
eval=/home/sagoyal/research/mdlm_refine/diffusion-vs-ar/eval_results/reconstruct/sudoku/refine
mkdir -p $eval

dataset=sudoku_test
input_file=/home/sagoyal/research/mdlm_refine/diffusion-vs-ar/data/${dataset}.jsonl

CUDA_VISIBLE_DEVICES=0 \
python3 -u eval_reconstruct.py \
    --model_name_or_path model_config_tiny \
    --checkpoint_dir $exp \
    --cache_dir /nfs/turbo/coe-jjparkcv-medium/satyam/.cache \
    --input_file $input_file \
    --dataset $dataset \
    --cutoff_len 164 \
    --sampler refine \
    --diffusion_steps 4 \
    --mask_frac 0.25 \
    --corrupt_frac 0.25 \
    --n_samples 100 \
    --eval_batch_size 10 \
    --n_correct_per_step 5 \
    --correct_mode topk \
    --proseco_budget 1 \
    --proseco_freq 1 \
    --save_gen_dir $eval/gen \
    --save_res_dir $eval/res \
    > $eval/${dataset}_refine_steps4.log
