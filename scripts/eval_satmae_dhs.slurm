#!/bin/bash
#SBATCH --job-name=eval_satmae_dhs
#SBATCH --partition=magma
#SBATCH --account=math022462
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/eval_satmae_dhs_%j.out
#SBATCH --error=logs/eval_satmae_dhs_%j.err


# Set up environment
export PYTHONPATH="/user/work/cy23765/kidsat_test/KidSat:$PYTHONPATH"

# Create logs directory if it doesn't exist
mkdir -p logs

# Navigate to your project directory
cd /user/work/cy23765/kidsat_test/KidSat


# Run the evaluation script
echo "Starting SatMAE DHS evaluation at $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"

python -c "
import sys
print('Python path:', sys.path)
print('Current working directory:', os.getcwd() if 'os' in globals() else 'os not imported')
"

# Run the evaluation
python -m modelling.satmae.eval_dhs

echo "Evaluation completed at $(date)"