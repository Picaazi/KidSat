#!/bin/bash
#SBATCH --job-name=train_satmae_%a
#SBATCH --partition=magma
#SBATCH --account=math022462
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=1-5
#SBATCH --output=logs/train_satmae_%A_%a.out
#SBATCH --error=logs/train_satmae_%A_%a.err


export PYTHONPATH="/user/work/cy23765/kidsat_test/KidSat:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

mkdir -p /user/work/cy23765/kidsat_test/KidSat/logs

cd /user/work/cy23765/kidsat_test/KidSat

# First download SatMAE checkpoints (you need to do this once)
# Download from the links mentioned in the repository
# Set the checkpoint path (change to your pretrained model path)
CHECKPOINT_PATH="pretrained_checkpoints/satmae/fmow_pretrain.pth"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT_PATH"
    echo "Please download SatMAE checkpoint first"
    exit 1
fi

# TRAINING: Fine-tune SatMAE
python -m modelling.satmae.satmae_finetune \
    --pretrained_ckpt $CHECKPOINT_PATH \
    --dhs_path survey_processing/processed_data/train_fold_${SLURM_ARRAY_TASK_ID}.csv \
    --output_path model_checkpoints/satmae/satmae_fold_${SLURM_ARRAY_TASK_ID} \
    --imagery_path data/imagery

echo "SatMAE training fold $SLURM_ARRAY_TASK_ID completed"