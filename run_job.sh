#!/bin/bash
#SBATCH --job-name=cascade_$(date +%Y%m%d)
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --time=12:00:00               
#SBATCH --mem=32G                     
#SBATCH --cpus-per-task=4             
#SBATCH --gpus-per-node=1             
#SBATCH --account=def-uanazodo-ab 

echo "Starting MRI Reconstruction Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

# 1. Copy the data into the node's fast local scratch space
echo "Copying dataset to SLURM_TMPDIR..."
cp ~/scratch/MRI_DATASET/data/k-space_data/*.pt $SLURM_TMPDIR/
echo "Data copied successfully!"

# 2. Load Python and activate environment
module load StdEnv/2023
module load python/3.13
module load cuda/13.2
source ~/CascadeNet_Cross_validation/ENV/bin/activate

# 3. Run the training script
python main_cv.py


echo "Job finished successfully."