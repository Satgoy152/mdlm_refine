#!/bin/bash
#SBATCH --job-name=train_sudoku_p
#SBATCH --output=slurm_output/train_sudoku_p.out
#SBATCH --error=slurm_output/train_sudoku_p.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition=spgpu2
#SBATCH --account=jjparkcv_owned1
#SBATCH --time=24:00:00
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=8

# Activate your venv
source /home/sagoyal/research/diffusion-vs-ar/.venv/bin/activate

export WANDB_DISABLED=false
export WANDB_PROJECT="diffusion-vs-ar-sudoku"


exp=/nfs/turbo/coe-jjparkcv-medium/satyam/sr/sudoku/proseco
eval=/home/sagoyal/research/diffusion-vs-ar/eval_results/sudoku/proseco
mkdir -p $exp
mkdir -p $eval


CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 2 --main_process_port 20099 \
src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir /nfs/turbo/coe-jjparkcv-medium/satyam/.cache \
    --model_name_or_path model_config_tiny \
    --do_train \
    --dataset sudoku_train \
    --finetuning_type full \
    --cutoff_len 164 \
    --output_dir $exp \
    --overwrite_cache \
    --per_device_train_batch_size 256 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --val_size 448 \
    --per_device_eval_batch_size 32 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_steps 500 \
    --learning_rate 1e-3 \
    --num_train_epochs 300.0 \
    --plot_loss \
    --run_name sr_proseco \
    --report_to wandb \
    --preprocessing_num_workers 8 \
    --fp16 \
    --save_total_limit 1 \
    --remove_unused_columns False \
    --diffusion_steps 20 \
    --save_safetensors False \
    --token_reweighting False \
    --time_reweighting none \
    --topk_decoding False \
    --alpha 0.25 \
    --gamma 1 \
    --refine True \
    --proseco True \
    --refine_remask_ratio t \
    --refine_temperature 0.0 \
    --refine_loss_type sum \
    > $exp/train.log || exit 1

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
    --diffusion_steps 20 \
    --output_dir $exp/${dataset} \
    --checkpoint_dir $exp  \
    --remove_unused_columns False \
    --decoding_strategy stochastic0.5-linear \
    --topk_decoding $topk_decoding \
    > $eval/${dataset}/eval-TopK$topk_decoding.log
done
