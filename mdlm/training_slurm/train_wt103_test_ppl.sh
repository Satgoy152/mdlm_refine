#!/bin/bash
#SBATCH --job-name=train_mdlm_vanilla
#SBATCH --output=train_mdlm_vanilla.out
#SBATCH --error=train_mdlm_vanilla.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --partition=spgpu2
#SBATCH --account=jjparkcv_owned1
#SBATCH --time=24:00:00
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=16


# Activate your venv
source /home/sagoyal/research/mdlm/.venv/bin/activate

srun python main.py \
  mode=ppl_eval \
  model=small \
  data=wikitext103 \
  data.cache_dir=/nfs/turbo/coe-jjparkcv-medium/satyam/.cache/datasets \
  backbone=dit \
  eval.checkpoint_path="/home/sagoyal/research/mdlm/outputs/openwebtext-train/2026.02.20/232048/checkpoints/best.ckpt" \
  parameterization=subs \
  model.length=1024 \
  wandb.name=mdlm-owt-finetuned-wt103-r-0.5-50k-testppl \
  +wandb.offline=true \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  loader.global_batch_size=64 \
  loader.num_workers=4 \
  training.refine=True \

