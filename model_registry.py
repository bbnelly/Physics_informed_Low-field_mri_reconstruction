# model_registry.py
"""Single source of truth for model architectures and hyperparameters.

Previously this dict was duplicated in main_cv.py, evaluate.py, and
evaluate_2.py — hyperparameters had to be kept in sync manually. Import
from here instead.
"""
from models import CascadeNet, DUNDD, UNetBaseline, MoDL, E2EVarNet

model_factories = {
    'UNet': lambda: UNetBaseline(features=32),
    'CascadeNet': lambda: CascadeNet(num_cascades=5, features=32),
    'DUNDD': lambda: DUNDD(num_iterations=5, lambda_dc=0.5, num_channels=64),
    'MoDL': lambda: MoDL(num_iterations=8, num_cg_steps=6, lambda_reg=0.05),
    'E2EVarNet': lambda: E2EVarNet(num_cascades=8, features=32),
}
