# model_registry.py
"""Single source of truth for model architectures and hyperparameters.

Previously this dict was duplicated in main_cv.py, evaluate.py, and
evaluate_2.py — hyperparameters had to be kept in sync manually. Import
from here instead.
"""
from models_3d import CascadeNet3D, DUNDD3D, E2EVarNet3D, MoDL3D, UNet3DBaseline

model_factories = {
    # Option A: full-volume 3D k-space reconstruction. These names are preserved
    # so existing experiment scripts continue to work, but they now instantiate
    # 3D architectures expecting tensors shaped (B, 2, kz, ky, kx).
    'UNet': lambda: UNet3DBaseline(features=16),
    'CascadeNet': lambda: CascadeNet3D(num_cascades=5, features=24),
    'DUNDD': lambda: DUNDD3D(num_iterations=5, lambda_dc=0.5, num_channels=32),
    'MoDL': lambda: MoDL3D(num_iterations=6, num_cg_steps=5, lambda_reg=0.05, features=32),
    'E2EVarNet': lambda: E2EVarNet3D(num_cascades=6, features=16),
}
