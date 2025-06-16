#!/bin/bash
#SBATCH --job-name=train_dino_spatial_%a
#SBATCH --partition=magma
#SBATCH --account=math022462
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=1-5  # Repository uses folds 1-5, not 0-4
#SBATCH --output=logs/train_dino_spatial_%A_%a.out
#SBATCH --error=logs/train_dino_spatial_%A_%a.err

echo "Start time: $(date)"

# Set up environment
export PYTHONPATH="/user/work/$USER/KidSat:$PYTHONPATH"

# Create directories
mkdir -p /user/work/$USER/KidSat/logs
mkdir -p /user/work/$USER/KidSat/model_checkpoints/dino_spatial

cd /user/work/$USER/KidSat

# TRAINING: Fine-tune DINOv2 for spatial prediction
python modelling/dino/finetune_spatial.py \
    --fold $SLURM_ARRAY_TASK_ID \
    --model_name dinov2_vitb14 \
    --imagery_path data/imagery \
    --imagery_source L \
    --batch_size 8 \
    --num_epochs 20 \

echo "Job completed at: $(date)"
echo "DINOv2 spatial training fold $SLURM_ARRAY_TASK_ID completed"

