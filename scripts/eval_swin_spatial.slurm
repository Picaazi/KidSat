#!/bin/bash
#SBATCH --job-name=eval_swin_spatial_%a
#SBATCH --partition=magma
#SBATCH --account=math022462
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_swin_spatial_%A_%a.out
#SBATCH --error=logs/eval_swin_spatial_%A_%a.err

echo "Start time: $(date)"

# Set up environment
export PYTHONPATH="/user/work/cy23765/kidsat_test/KidSat:$PYTHONPATH"


# Create directories
mkdir -p /user/work/cy23765/kidsat_test/KidSat/results/swin_spatial

cd /user/work/cy23765/kidsat_test/KidSat

# Evaluating swin spatial
python -m modelling.dino.evaluate_swin\
    --model_name swin_b \
    --target deprived_sev \
    --imagery_source L \
    --imagery_path data/imagery \
    --mode spatial \
    --use_checkpoint \
    --emb_size 1000 \
    > results/swin_spatial/evaluation.log 2>&1

echo "Job completed at: $(date)"
echo "Swin spatial completed"

