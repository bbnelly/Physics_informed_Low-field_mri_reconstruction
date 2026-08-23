# evaluate.py
import os
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
from config import OUTPUT_DIR, TEST_SUBJECTS
from data_loader import load_and_separate_dataset, ValMRIDataset
from models import CascadeNet, DUNDD, UNetBaseline, MoDL, E2EVarNet
from train import run_training
from visualize import generate_reconstruction_pdf, plot_training_curves, plot_comparison_table

# Model mapping (must match train.py)
model_factories = {
    'UNet': lambda: UNetBaseline(features=32),
    'CascadeNet': lambda: CascadeNet(num_cascades=5, features=32),
    'DUNDD': lambda: DUNDD(num_iterations=5, lambda_dc=0.5, num_channels=64),
    'MoDL': lambda: MoDL(num_iterations=8, num_cg_steps=6, lambda_reg=0.05),
    'E2EVarNet': lambda: E2EVarNet(num_cascades=8, features=32),
}


def evaluate_all_models(models_to_run, train_df, val_df, test_df, num_epochs=50, batch_size=8):
    """Train all models and generate comparison PDF."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}
    trained_models = {}

    for model_name, model_fn in models_to_run.items():
        print(f"\n{'='*60}\nTraining: {model_name}\n{'='*60}")
        model = model_fn()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")

        trained_model, history = run_training(
            train_df, val_df, model, model_name=model_name,
            num_epochs=num_epochs, batch_size=batch_size, device=device
        )

        best_ssim = max(history['val_ssim'])
        best_psnr = max(history['val_psnr'])
        best_ep = history['val_ssim'].index(best_ssim) + 1

        all_results[model_name] = {
            'best_ssim': best_ssim, 'best_psnr': best_psnr,
            'best_epoch': best_ep, 'n_params': n_params,
            'all_val_ssim': history['val_ssim'],
            'all_val_psnr': history['val_psnr'],
            'all_train_loss': history['train_loss'],
        }
        trained_models[model_name] = trained_model

        # Save history
        with open(os.path.join(OUTPUT_DIR, f'{model_name}_history.json'), 'w') as f:
            json.dump(history, f, indent=2)

        del trained_model
        torch.cuda.empty_cache()

    # Generate visualizations
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)

    # Training curves for each model
    for name, res in all_results.items():
        # Reconstruct history from saved data
        history = {'train_loss': res['all_train_loss'],
                   'val_psnr': res['all_val_psnr'],
                   'val_ssim': res['all_val_ssim']}
        plot_training_curves(history, save_path=os.path.join(OUTPUT_DIR, f'curves_{name}.png'))

    # Comparison table
    plot_comparison_table(all_results)

    # Load models for PDF generation
    loaded_models = {}
    for name in models_to_run.keys():
        ckpt_path = os.path.join('checkpoints', f'best_{name}.pt')
        if os.path.exists(ckpt_path):
            model = model_factories[name]()
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state'])
            model.eval()
            model.to(device)
            loaded_models[name] = model
            print(f"Loaded {name} for PDF generation")

    if loaded_models:
        # Create validation loader for visualization (use test data)
        test_loader = DataLoader(ValMRIDataset(test_df), batch_size=8, shuffle=False)

        generate_reconstruction_pdf(
            loaded_models, test_loader, device, all_results,
            pdf_path=os.path.join(OUTPUT_DIR, 'reconstruction_comparison.pdf'),
            n_samples=20
        )

    return all_results


def evaluate_cross_validation(model_name='CascadeNet', num_epochs=70):
    """Run leave-one-subject-out cross-validation and generate visualizations."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    df, fully_sampled_df, _ = load_and_separate_dataset()

    # CV subjects (all except held-out test subjects)
    all_subjects = list(fully_sampled_df['subject'].unique())
    cv_subjects = [s for s in all_subjects if s not in TEST_SUBJECTS]
    cv_df = fully_sampled_df[fully_sampled_df['subject'].isin(cv_subjects)].reset_index(drop=True)

    cv_results = {}
    for fold_idx, val_subject in enumerate(cv_subjects):
        print(f"\nFOLD {fold_idx+1}/{len(cv_subjects)} | Model: {model_name} | Val: {val_subject}")
        fold_train_df = cv_df[cv_df['subject'] != val_subject].reset_index(drop=True)
        fold_val_df = cv_df[cv_df['subject'] == val_subject].reset_index(drop=True)

        model = model_factories[model_name]()
        _, history = run_training(
            fold_train_df, fold_val_df, model,
            model_name=f"{model_name}_fold{fold_idx+1}",
            num_epochs=num_epochs, device=device
        )

        best_ssim = max(history['val_ssim'])
        best_psnr = max(history['val_psnr'])
        cv_results[val_subject] = {'best_ssim': best_ssim, 'best_psnr': best_psnr}

        del model
        torch.cuda.empty_cache()

    # Print summary
    print("\n" + "="*60)
    print(f"CROSS-VALIDATION RESULTS — {model_name}")
    print("="*60)
    all_ssims = [r['best_ssim'] for r in cv_results.values()]
    all_psnrs = [r['best_psnr'] for r in cv_results.values()]
    print(f"Mean SSIM: {np.mean(all_ssims):.4f} ± {np.std(all_ssims):.4f}")
    print(f"Mean PSNR: {np.mean(all_psnrs):.2f} ± {np.std(all_psnrs):.2f}")

    # Save CV results
    with open(os.path.join(OUTPUT_DIR, f'cv_results_{model_name}.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)

    return cv_results


if __name__ == "__main__":
    # Example: Run full evaluation for all models
    df, fully_sampled_df, _ = load_and_separate_dataset()

    train_df = fully_sampled_df[fully_sampled_df['subject'].isin(['9092', '9133', '9147', '9139', '9110'])].reset_index(drop=True)
    val_df = fully_sampled_df[fully_sampled_df['subject'].isin(['9101', '9074'])].reset_index(drop=True)
    test_df = fully_sampled_df[fully_sampled_df['subject'].isin(TEST_SUBJECTS)].reset_index(drop=True)

    models_to_run = {
        'UNet': lambda: UNetBaseline(features=32),
        'CascadeNet': lambda: CascadeNet(num_cascades=5, features=32),
    }

    results = evaluate_all_models(models_to_run, train_df, val_df, test_df, num_epochs=50)