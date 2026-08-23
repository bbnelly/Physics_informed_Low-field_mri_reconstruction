#!/bin/bash
#SBATCH --job-name=cascade_cv
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --account=def-uanazodo-ab

echo "Starting MRI Reconstruction Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

# NOTE: the old "cp *.pt $SLURM_TMPDIR/" step was removed — it copied files
# that were never read. config.py's BASE_PATH points directly at the network
# scratch path, and data_loader.py reads .h5 files, not .pt.

# Copy the .h5 data tree to fast local scratch and point the pipeline at it.
echo "Copying dataset to SLURM_TMPDIR..."
if cp -r ~/scratch/MRI_DATASET/data $SLURM_TMPDIR/ 2>/dev/null; then
    export MRI_DATA_PATH=$SLURM_TMPDIR/data
    echo "Using local scratch data: $MRI_DATA_PATH"
else
    echo "⚠️  Data copy failed — falling back to ~/scratch path"
fi

module load StdEnv/2023
module load python/3.13
source ~/CascadeNet_Cross_validation/ENV/bin/activate

echo "Checking torch/CUDA..."
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())" \
  || { echo "Torch/CUDA check failed"; exit 1; }

# Set RESUME=1 as an environment variable when submitting to resume the most
# recent run for this model instead of starting fresh, e.g.:
#   sbatch --export=RESUME=1 run_job.sh
RESUME_FLAG=""
if [ "$RESUME" = "1" ]; then
    echo "Resuming most recent run..."
    RESUME_FLAG="--resume"
fi

python main_cv.py --model CascadeNet --epochs 2 $RESUME_FLAG
MAIN_CV_EXIT=$?

if [ $MAIN_CV_EXIT -eq 0 ]; then
    echo "Running evaluation suite..."
    python evaluate_2.py --task all --model CascadeNet
else
    echo "main_cv.py failed (exit $MAIN_CV_EXIT) — skipping evaluation"
fi

echo "Job finished."