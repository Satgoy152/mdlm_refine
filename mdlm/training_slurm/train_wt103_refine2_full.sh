#!/bin/bash
#SBATCH --job-name=train_mdlm_refine2
#SBATCH --output=slurm_output/train_mdlm_refine2_full.out
#SBATCH --error=slurm_output/train_mdlm_refine2_full.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=spgpu2
#SBATCH --account=jjparkcv_owned1
#SBATCH --time=48:00:00
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=16

nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

variant=refine_2
# Activate your venv
source /home/sagoyal/research/mdlm_refine/mdlm/.venv/bin/activate

python main.py \
  mode=train_eval \
  model=small \
  data=openwebtext-split \
  data.cache_dir=/nfs/turbo/coe-jjparkcv-medium/satyam/.cache/datasets \
  backbone=dit \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  checkpointing.resume_from_ckpt=true \
  checkpointing.resume_ckpt_path="/home/sagoyal/research/mdlm_refine/mdlm/checkpoints/mdlm.ckpt" \
  parameterization=subs \
  model.length=1024 \
  wandb.name=${variant} \
  wandb.id="${variant}_$(date +%Y%m%d_%H%M%S)" \
  loader.eval_batch_size=8 \
  loader.global_batch_size=48 \
  sampling.steps=1000 \
  loader.num_workers=2 \
  trainer.val_check_interval=1000 \
  trainer.max_steps=1172551 \
  training.refine=True \
  training.refine_variant=${variant} \
  training.remask_ratio='t' \
  training.loss_type="alpha" \
  training.alpha=0.5 \
  checkpointing.save_dir=/nfs/turbo/coe-jjparkcv-medium/satyam/mdlm/${variant} \

