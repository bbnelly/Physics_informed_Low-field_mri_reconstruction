#!/bin/bash
#SBATCH --job-name=mri_rerun
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

# 1. Copy the .h5 data tree to the node's fast local scratch space.
#    (The pipeline reads .h5 files via config.BASE_PATH; old *.pt copies were never used.)
echo "Copying dataset to SLURM_TMPDIR..."
if cp -r ~/scratch/MRI_DATASET/data $SLURM_TMPDIR/ 2>/dev/null; then
    export MRI_DATA_PATH=$SLURM_TMPDIR/data
    echo "Using local scratch data: $MRI_DATA_PATH"
else
    echo "⚠️  Data copy failed — falling back to ~/scratch path"
fi

# 2. Load Python and activate environment (same env as root run_job.sh)
module load StdEnv/2023
module load python/3.13
source ~/CascadeNet_Cross_validation/ENV/bin/activate

# 3. Run the requested model as a fresh tagged rerun.
MODEL="${MODEL:-CascadeNet}"
RERUN_TAG="${RERUN_TAG:-lr5e-4_rerun}"
export RERUN_TAG

# 4. Sanity-check torch/CUDA before training
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())" \
  || { echo "Torch/CUDA check failed"; exit 1; }

# 5. Run the training script (run from repo root so relative paths work)
cd ~/CascadeNet_Cross_validation
python main_cv.py --model "$MODEL" --epochs 40 --batch_size 8 --acceleration 2
MAIN_CV_EXIT=$?

if [ "$MAIN_CV_EXIT" -eq 0 ]; then
    python evaluate_2.py --task all --model "$MODEL"
else
    echo "main_cv.py failed (exit $MAIN_CV_EXIT) - skipping evaluation"
fi

echo "Job finished successfully."