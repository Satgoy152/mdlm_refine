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

task=sudoku
sampling=proseco
model=proseco
exp=/nfs/turbo/coe-jjparkcv-medium/satyam/sr/${task}/${model}
eval=/home/sagoyal/research/mdlm_refine/diffusion-vs-ar/eval_results/${task}/${model}
mkdir -p $eval

samples=1000
steps=2

dataset=${task}_test
input_file=/home/sagoyal/research/mdlm_refine/diffusion-vs-ar/data/${dataset}.csv

CUDA_VISIBLE_DEVICES=0 \
python3 -u eval_reconstruct.py \
    --model_name_or_path model_config \
    --checkpoint_dir $exp \
    --cache_dir /nfs/turbo/coe-jjparkcv-medium/satyam/.cache \
    --input_file $input_file \
    --dataset $dataset \
    --cutoff_len 164 \
    --sampler $sampling \
    --topk_decoding \
    --decoding_strategy deterministic-linear \
    --diffusion_steps $steps \
    --mask_frac 0.0 \
    --corrupt_frac 1.0 \
    --n_samples $samples \
    --eval_batch_size 20 \
    --n_correct_per_step 16 \
    --correct_mode topk \
    --seed 45 \
    --verbose \
    --proseco_budget 1 \
    --proseco_freq 1 \
    --save_gen_dir $eval/gen \
    --save_res_dir $eval/res \
    2>&1 | tee $eval/${sampling}/samples${samples}_steps${steps}.log
