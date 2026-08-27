# 3D MRI Reconstruction Experiments

This project trains and evaluates 3D MRI k-space reconstruction models using deterministic leave-one-subject-out (LOSO) cross-validation. It supports five model architectures, acceleration-factor sweeps, reconstruction metrics, and saved experiment artifacts.

## Models

The model registry contains:

- `CascadeNet`
- `DUNDD`
- `E2EVarNet`
- `MoDL`
- `UNet`

All models operate on full 3D volumes represented as two-channel real/imaginary k-space tensors.

## Requirements

Use Python 3.10+ and install the dependencies from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On the current compute environment, the existing shared environment can also be activated:

```bash
source /home/shadow/CascadeNet_Cross_validation/ENV/bin/activate
```

## Dataset

By default, the code reads MRI data from:

```text
~/scratch/MRI_DATASET/data
```

Set `MRI_DATA_PATH` to override this location:

```bash
export MRI_DATA_PATH=/path/to/MRI_DATASET/data
```

The data loader expects fully sampled MRI files in the configured dataset directory and loads them into 3D volumes. The current configuration uses:

- 38 kz/slice partitions
- 136 phase-encode samples
- 150 readout samples
- LOSO subjects: `9033`, `9070`, `9074`, `9092`, `9101`, `9110`, `9133`, `9139`, `9147`

## Train With Cross-Validation

Run LOSO cross-validation for one model:

```bash
python main_cv.py \
  --model CascadeNet \
  --epochs 60 \
  --batch_size 1 \
  --acceleration 2
```

Supported options include:

- `--model`: one registered model name
- `--epochs`: number of training epochs
- `--batch_size`: validation/training batch size
- `--acceleration`: undersampling acceleration factor
- `--resume`: resume an existing run where supported
- `--run_id`: target a specific saved run

For a full experiment plan across all five models:

```bash
python run_experiment.py --phase baseline_cv --epochs 60 --batch_size 1 --acceleration 2
```

The SLURM submission script can be adapted for cluster execution when GPU or scheduled compute resources are needed.

## Evaluate A Saved Run

Evaluation operates on a specific run directory under:

```text
~/scratch/MRI_DATASET/Nelson_runs/<run_id>/
```

Run an acceleration sweep for a saved model run:

```bash
python evaluate_2.py \
  --task sweep \
  --model CascadeNet \
  --run_id CascadeNet_<timestamp>_job<id>
```

The standard sweep uses acceleration factors:

```text
R = 2, 4, 6, 8, 10, 12
```

Other evaluation tasks are available through `--task`:

```text
sweep, boxplot, stats, visualize, all
```

## Generate Figures And Tables

The figure generator uses the configured saved runs and writes to `main results` by default:

```bash
python generate_result_figures.py \
  --models CascadeNet DUNDD E2EVarNet MoDL UNet \
  --slice 27 \
  --output-dir "/home/shadow/scratch/MRI_DATASET/Nelson_runs/main results"
```

This generates:

- One 3x2 acceleration-sweep image per model at the selected slice
- A five-model acceleration comparison plot
- Reconstruction comparison figures
- Training-loss plots when fold histories are available
- Long-form acceleration sweep CSV/Markdown results
- Five-row mean PSNR and SSIM tables
- A black-and-white PSNR/SSIM acceleration-sweep summary graph

The mean summary artifacts are:

```text
acceleration_sweep_mean_psnr.csv
acceleration_sweep_mean_psnr.md
acceleration_sweep_mean_ssim.csv
acceleration_sweep_mean_ssim.md
acceleration_sweep_summary_black_white.png
```

Each mean table has one row per model, columns for `R=2`, `R=4`, `R=6`, `R=8`, `R=10`, `R=12`, and a mean across those acceleration factors.

## Metrics

Metrics are computed in the image-magnitude domain after inverse FFT reconstruction:

- PSNR in dB
- SSIM

Predictions and targets are normalized using the target image range before metric calculation. The acceleration sweep stores both model and zero-filled baseline metrics.

## Project Layout

```text
config.py                    Dataset and training defaults
data_loader.py              MRI loading and validation datasets
masks.py                    3D ky-kz undersampling masks
models_3d.py                3D model implementations
model_registry.py           Model name-to-architecture registry
train.py                    Training and checkpoint utilities
main_cv.py                  LOSO cross-validation entry point
evaluate_2.py               Post-training evaluation suite
generate_result_figures.py  Figure and summary-table generation
run_manager.py              Saved run path and metadata management
run_experiment.py           Multi-phase experiment runner
requirements.txt            Python dependencies
checkpoints/                Local checkpoints, when used
outputs/                    Local generated outputs, when used
```

Saved cluster runs are organized separately under `~/scratch/MRI_DATASET/Nelson_runs`, with `checkpoints/`, `results/`, `figures/`, and `logs/` subdirectories.

## Reproducibility Notes

- Use the same dataset path and run ID when comparing results.
- Keep the model name consistent with the checkpoint filename.
- Use saved `cv_results_<model>.json` and `reliability_sweep_<model>.json` files for post-hoc plots and tables.
- Check the run metadata and logs alongside checkpoints when reproducing an experiment.
