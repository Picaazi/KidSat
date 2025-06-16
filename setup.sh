#!/bin/bash
# KidSat Project Setup Script for BluePebble HPC
# Created: $(date)
# Usage: source ~/setup.sh

echo "Setting up KidSat environment on BluePebble..."

# Load required modules
echo " ^=^s  Loading modules..."
module load languages/python/3.8.20

# Set environment variables
echo " ^=^t  Setting environment variables..."
export PYTHONPATH="/user/work/$USER/KidSat:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# Navigate to project directory (change to your working directory)
cd /user/work/$USER/kidsat_test/KidSat

# Create necessary directories
echo " ^=^s^a Creating project directories..."
mkdir -p logs
mkdir -p results
mkdir -p model_checkpoints/dino_spatial
mkdir -p model_checkpoints/dino_temporal
mkdir -p model_checkpoints/satmae
mkdir -p scripts

# Install Python dependencies (first time only)
if [ ! -f ".dependencies_installed" ]; then
    echo " ^=^p^m Installing Python dependencies..."
    pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install --user --upgrade pip setuptools wheel
    pip install --user pyproj
    
    # Install packages in the correct order (dependencies first)
    pip install --user \ "pandas==1.3.5" \ "fiona==1.8.21" \ "shapely==1.8.5" \ 
    "pyproj==3.2.1" \ "geopandas==0.10.2" \ "earthengine-api" \
    "tqdm"

    pip install --user transformers
    pip install --user timm
    pip install --user scikit-learn
    pip install --user matplotlib
    pip install --user seaborn
    pip install --user rasterio
    pip install --user Pillow
    
    # Mark dependencies as installed
    touch .dependencies_installed
    echo " ^|^e Dependencies installed successfully"
else
    echo " ^|^e Dependencies already installed"
fi

# Check data structure
echo " ^=^s^j Checking data structure..."
if [ -d "survey_processing/processed_data" ]; then
    echo " ^|^e DHS processed data found"
    ls survey_processing/processed_data/*.csv | head -5
else
    echo " ^z   ^o  DHS processed data not found. Please run get_imagery.ipynb first."
fi

if [ -d "data/imagery" ]; then
    echo " ^|^e Satellite imagery directory found"
    echo "   Total subdirectories: $(find data/imagery -type d -mindepth 1 | wc -l)"
    echo "   Total .tif files: $(find data/imagery -name "*.tif" | wc -l)"
else
    echo " ^z   ^o  Satellite imagery not found. Please upload your imagery data."
fi

# Display system info
echo " ^=^r  System Information:"
echo "   Current directory: $(pwd)"
echo "   Python version: $(python --version)"
echo "   PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"

echo " ^=^n^i Setup complete! You can now submit training jobs."
echo ""
echo " ^=^s^} Quick commands:"
echo "   Check job status: squeue -u $USER"
echo "   Submit training: sbatch scripts/train_dino_spatial.slurm"
echo "   Monitor logs: tail -f logs/train_dino_spatial_*.out"