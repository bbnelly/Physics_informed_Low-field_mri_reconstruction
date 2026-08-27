#!/bin/bash
#SBATCH --job-name=unet_cv
#SBATCH --output=training_log_UNet_%j.out
#SBATCH --error=training_log_UNet_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --account=def-uanazodo-ab

echo "Starting MRI Reconstruction Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

# 1. Copy the dataset into the node's fast local scratch space
echo "Copying dataset to SLURM_TMPDIR..."
if cp -r ~/scratch/MRI_DATASET/data $SLURM_TMPDIR/ 2>/dev/null; then
    export MRI_DATA_PATH=$SLURM_TMPDIR/data
    echo "Using local scratch data: $MRI_DATA_PATH"
else
    echo "Data copy failed - falling back to ~/scratch path"
fi

# 2. Load Python and activate environment
module load StdEnv/2023
module load python/3.13
source ~/CascadeNet_Cross_validation/ENV/bin/activate

# 3. Run cross-validation training
python main_cv.py --model UNet --epochs 40 ${RESUME:+--resume}
MAIN_CV_EXIT=$?

# 4. Run evaluation only if training succeeded
if [ $MAIN_CV_EXIT -eq 0 ]; then
    echo "Running evaluation suite..."
    python evaluate_2.py --task all --model UNet
else
    echo "main_cv.py failed (exit $MAIN_CV_EXIT) - skipping evaluation"
fi

echo "Job finished."