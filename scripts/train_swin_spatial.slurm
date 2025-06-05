#!/bin/bash
#SBATCH --job-name=train_swin_spatial_%a
#SBATCH --partition=magma
#SBATCH --account=math022462
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=1-5  # Repository uses folds 1-5, not 0-4
#SBATCH --output=logs/train_swin_spatial_%A_%a.out
#SBATCH --error=logs/train_swin_spatial_%A_%a.err

echo "Start time: $(date)"

# Set up environment
export PYTHONPATH="/user/work/cy23765/kidsat_test/KidSat:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# Create directories
mkdir -p /user/work/cy23765/kidsat_test/KidSat/logs
mkdir -p /user/work/cy23765/kidsat_test/KidSat/model_checkpoints/swin_spatial

cd /user/work/cy23765/kidsat_test/KidSat

# TRAINING: Fine-tune swin for spatial prediction
python modelling/dino/finetune_spatial_swin.py \
    --fold $SLURM_ARRAY_TASK_ID \
    --model_name swin_b \
    --imagery_path data/imagery \
    --imagery_source L \
    --batch_size 8 \
    --num_epochs 20 

echo "Job completed at: $(date)"
echo "Swin spatial training fold $SLURM_ARRAY_TASK_ID completed"

