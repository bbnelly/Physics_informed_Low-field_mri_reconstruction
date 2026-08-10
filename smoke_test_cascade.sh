#!/bin/bash
#SBATCH --job-name=cascade_smoketest
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --time=00:50:00               
#SBATCH --mem=32G                     
#SBATCH --cpus-per-task=4             
#SBATCH --gpus-per-node=1             
#SBATCH --account=def-uanazodo-ab 

echo "Starting MRI Reconstruction Job (SMOKE TEST)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

# 1. Copy the data into the node's fast local scratch space
echo "Copying dataset to SLURM_TMPDIR..."
cp ~/scratch/MRI_DATASET/data/k-space_data/*.pt $SLURM_TMPDIR/
echo "Data copied successfully!"

# 2. Load Python and activate environment
module load StdEnv/2023
module load python/3.13
# cuda module intentionally omitted for this test — the wheelhouse
# torch build should bundle its own CUDA runtime. Re-add
# `module load cuda/13.2` above this line if the check below fails.
source ~/CascadeNet_Cross_validation/ENV/bin/activate

# 3. Sanity-check torch/CUDA before touching real training code
echo "Checking torch/CUDA..."
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())" \
  || { echo "Torch/CUDA check failed"; exit 1; }

# 4. Run the training script
#    If main_cv.py supports an epoch-limiting flag, add it here, e.g.:
#    python main_cv.py --epochs 1
python main_cv.py

echo "Job finished successfully."
