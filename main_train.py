# main_train.py
import torch
from config import DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE
from data_loader import load_and_separate_dataset
from models import CascadeNet
from train import run_training

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    df, fully_sampled_df, _ = load_and_separate_dataset()

    # Check what subjects are available
    available_subjects = sorted(fully_sampled_df['subject'].unique())
    print(f"Available subjects in fully sampled: {available_subjects}")

    # Use all available subjects for testing
    train_df = fully_sampled_df.sample(frac=0.7, random_state=42).reset_index(drop=True)
    val_df = fully_sampled_df.drop(train_df.index).reset_index(drop=True)

    print(f"Train: {len(train_df)} files")
    print(f"Val: {len(val_df)} files")

    model = CascadeNet(num_cascades=3, features=16)

    _, history = run_training(
        train_df, val_df, model,
        model_name='CascadeNet',
        num_epochs=1,  # Test with 1 epoch
        batch_size=4,
        device=device
    )

    print(f"Training complete. Best SSIM: {max(history['val_ssim']):.4f}")

if __name__ == "__main__":
    main()