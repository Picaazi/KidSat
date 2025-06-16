#!/bin/bash
#SBATCH --job-name=train_sh_siren_spatial_TZ%a
#SBATCH --partition=magma
#SBATCH --account=math022462
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=1-5  # Repository uses folds 1-5, not 0-4
#SBATCH --output=logs/train_sh_siren_spatial_TZ%A_%a.out
#SBATCH --error=logs/train_sh_siren_spatial_TZ%A_%a.err

echo "Start time: $(date)"

# Set up environment
export PYTHONPATH="/user/work/cy23765/kidsat_test/KidSat:$PYTHONPATH"

# Create directories
mkdir -p /user/work/cy23765/kidsat_test/KidSat/logs
mkdir -p /user/work/cy23765/kidsat_test/KidSat/model_checkpoints/sh_siren_spatial

cd /user/work/cy23765/kidsat_test/KidSat

# TRAINING: Fine-tune Siren for spatial prediction
python modelling/dino/finetune_siren.py \
    --fold $SLURM_ARRAY_TASK_ID \
    --imagery_path data/imagery \
    --imagery_source L \
    --representation_dim 128 \
    --hidden_dim 256 \
    --num_layers 4 \
    --batch_size 32 \
    --num_epochs 200 \
    --sh_L 15 \
    --country TZ

echo "Job completed at: $(date)"
echo "Siren spatial training fold $SLURM_ARRAY_TASK_ID completed"

