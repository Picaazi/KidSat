#!/bin/bash
#SBATCH --job-name=eval_satmae_spatial_%a
#SBATCH --partition=magma
#SBATCH --account=math022462
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_satmae_spatial_%A_%a.out
#SBATCH --error=logs/eval_satmae_spatial_%A_%a.err


echo "Start time: $(date)"
export PYTHONPATH="/user/work/cy23765/kidsat_test/KidSat:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

mkdir -p /user/work/cy23765/kidsat_test/KidSat/results/satmae_spatial

cd /user/work/cy23765/kidsat_test/KidSat

# EVALUATION: Test the fine-tuned satmae model
python -m modelling.satmae.satmae_eval \
    --imagery_path data/imagery \
    --output_path results/satmae_spatial \
    > results/satmae_spatial/evaluation.log 2>&1


echo "Job completed at: $(date)"
echo "SatMAE spatial evaluation completed"