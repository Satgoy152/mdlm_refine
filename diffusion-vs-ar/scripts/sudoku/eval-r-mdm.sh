#!/bin/bash
#SBATCH --job-name=train_sudoku_r
#SBATCH --output=slurm_output/train_sudoku_r.out
#SBATCH --error=slurm_output/train_sudoku_r.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition=spgpu2
#SBATCH --account=jjparkcv_owned1
#SBATCH --time=24:00:00
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=8

# Activate your venv
# source /home/sagoyal/research/diffusion-vs-ar/.venv/bin/activate

export WANDB_DISABLED=true
export WANDB_PROJECT="diffusion-vs-ar"


exp=/nfs/turbo/coe-jjparkcv-medium/satyam/sr/sudoku/refine
eval=/home/sagoyal/research/diffusion-vs-ar/eval_results/sudoku/refine
mkdir -p $exp
mkdir -p $eval

for dataset in sudoku_test
do
topk_decoding=True
mkdir $eval/$dataset
CUDA_VISIBLE_DEVICES=0  \
python3 -u src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir /nfs/turbo/coe-jjparkcv-medium/satyam/.cache \
    --model_name_or_path model_config_tiny \
    --do_predict \
    --cutoff_len 164 \
    --dataset $dataset \
    --finetuning_type full \
    --diffusion_steps 4 \
    --output_dir $exp/${dataset} \
    --checkpoint_dir $exp  \
    --remove_unused_columns False \
    --decoding_strategy stochastic0.5-linear \
    --topk_decoding $topk_decoding \
    --sampling_method vanilla \
    --n_correct_per_step 5 \
    --correct_mode topk \
    > $eval/${dataset}/eval-TopK$topk_decoding.log
done