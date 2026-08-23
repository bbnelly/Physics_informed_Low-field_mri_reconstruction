"""Experiment schedule derived from the project plan in the design image.

The workflow is:
1. Baseline CV across all candidate models for 60 epochs.
2. Keep the top 2 models for deeper study.
3. Evaluate those models across acceleration factors R = 2, 4, 6, 8, 10, 12.
4. Train the best model at the best acceleration with longer epochs.
5. Tune hyperparameters on the top 2 models.

This file defines the structure but does not execute training directly. A small
runner can pick a phase and call the existing CV/evaluation entrypoints.
"""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ExperimentStep:
    name: str
    description: str
    models: list[str]
    epochs: int
    acceleration_factors: list[int] | None = None
    notes: str = ""


def get_experiment_plan() -> list[ExperimentStep]:
    return [
        ExperimentStep(
            name="baseline_cv",
            description="Train all candidate models with 5-fold/LOSO-style CV for 60 epochs to establish a baseline.",
            models=["CascadeNet", "UNet", "DUNDD", "MoDL", "E2EVarNet"],
            epochs=60,
            acceleration_factors=[2],
            notes="CascadeNet is the primary model focus, while E2EVarNet acts as the high-performance reference.",
        ),
        ExperimentStep(
            name="shortlist_top2",
            description="Select the two best-performing models based on validation SSIM/PSNR and continue with them only.",
            models=["CascadeNet", "E2EVarNet"],
            epochs=60,
            acceleration_factors=[2],
            notes="This step is model-selection-driven; it does not retrain the full suite once the shortlist is fixed.",
        ),
        ExperimentStep(
            name="acceleration_sweep",
            description="Evaluate the shortlisted models at multiple acceleration factors to see where performance degrades.",
            models=["CascadeNet", "E2EVarNet"],
            epochs=60,
            acceleration_factors=[2, 4, 6, 8, 10, 12],
            notes="This matches the reliability sweep in evaluate_2.py.",
        ),
        ExperimentStep(
            name="extended_training",
            description="Train the best model at the best acceleration setting for more epochs until convergence is observed.",
            models=["CascadeNet"],
            epochs=300,
            acceleration_factors=[2],
            notes="This is the longest-run convergence stage in the experiment plan.",
        ),
        ExperimentStep(
            name="hyperparameter_tuning",
            description="Tune the top-performing models on the best under-sampling technique and the most promising hyperparameters.",
            models=["CascadeNet", "E2EVarNet"],
            epochs=60,
            acceleration_factors=[2],
            notes="Use this stage for architecture and scheduler / learning-rate exploration after shortlist selection.",
        ),
    ]


def get_step_by_name(name: str) -> ExperimentStep:
    for step in get_experiment_plan():
        if step.name == name:
            return step
    raise ValueError(f"Unknown experiment step: {name}")


def describe_plan() -> list[dict]:
    return [asdict(step) for step in get_experiment_plan()]
