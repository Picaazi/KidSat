#!/bin/bash
#SBATCH --job-name=eval_dino_spatial_%a
#SBATCH --partition=magma
#SBATCH --account=math022462
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_dino_spatial_%A_%a.out
#SBATCH --error=logs/eval_dino_spatial_%A_%a.err



echo "Start time: $(date)"
export PYTHONPATH="/user/work/$USER/KidSat:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

mkdir -p /user/work/$USER/KidSat/results/dino_spatial

cd /user/work/$USER/KidSat

source setup.sh

# EVALUATION: Test the fine-tuned DINOv2 model
#         --grouped_bands 4 3 2\
python modelling/dino/evaluate.py \
    --imagery_source L \
    --imagery_path data/imagery \
    --mode spatial \
    --use_checkpoint \


    
    > results/dino_spatial/evaluation.log 2>&1

    


echo "Job completed at: $(date)"
echo "DINOv2 spatial evaluation completed"