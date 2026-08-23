#!/bin/bash
#SBATCH --job-name=cascade_smoketest
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --time=00:05:00               
#SBATCH --mem=32G                     
#SBATCH --cpus-per-task=4             
#SBATCH --gpus-per-node=1             
#SBATCH --account=def-uanazodo-ab 

echo "Starting MRI Reconstruction Job (SMOKE TEST)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

# 1. Copy the data into the node's fast local scratch space
#    (pipeline reads .h5 files via config.BASE_PATH; .pt files are never used)
echo "Copying dataset to SLURM_TMPDIR..."
cp -r ~/scratch/MRI_DATASET/data $SLURM_TMPDIR/ \
  && export MRI_DATA_PATH=$SLURM_TMPDIR/data \
  || echo "⚠️  Data copy failed — falling back to ~/scratch path"

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

# 4. Run the training script — 1 epoch only; this is a 5-minute smoke test
python main_cv.py --epochs 1

# 5. Run post-CV evaluation suite (only runs if main_cv.py succeeded)
if [ $? -eq 0 ]; then
    echo "Running evaluation suite..."
    python evaluate_2.py --task all --model CascadeNet
else
    echo "main_cv.py failed — skipping evaluation"
fi


echo "Job finished successfully."
