"""
run_manager.py

Gives every training or evaluation run a unique, timestamped identity and
routes ALL of its outputs (checkpoints, figures, results JSON, logs, and
uncaught error tracebacks) into one self-contained folder on scratch.

Usage in main_cv.py:

    from run_manager import setup_run

    run = setup_run(model_name="CascadeNet", base_dir=os.path.expanduser(
        "~/scratch/MRI_DATASET/runs"))

    # run.checkpoints, run.figures, run.results, run.logs are all Path objects,
    # already created on disk. run.run_id is the human-readable identifier.
    # run.save_config({...}) snapshots your hyperparameters/config as JSON.
    # Uncaught exceptions anywhere in the process are automatically logged
    # with full traceback to run.logs/run.log before the process exits.

Usage in evaluate.py (to point at a specific past run instead of creating
a new one):

    from run_manager import load_run

    run = load_run("CascadeNet_20260812-143005_job534012",
                    base_dir=os.path.expanduser("~/scratch/MRI_DATASET/runs"))
"""
import os
import sys
import json
import logging
import subprocess
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


def _get_slurm_job_id() -> str:
    """Returns the SLURM job ID if running under sbatch/salloc, else 'local'."""
    return os.environ.get("SLURM_JOB_ID", "local")


def _get_git_commit() -> str:
    """Best-effort short git commit hash for reproducibility. Returns 'unknown'
    if not in a git repo or git isn't available (e.g. some compute nodes)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _make_run_id(model_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_id = _get_slurm_job_id()
    tag = os.environ.get("RERUN_TAG", "").strip()
    tag_part = f"_{tag}" if tag else ""
    return f"{model_name}{tag_part}_{timestamp}_job{job_id}"


@dataclass
class RunPaths:
    run_id: str
    root: Path
    checkpoints: Path
    figures: Path
    results: Path
    logs: Path

    def save_config(self, config: dict) -> None:
        """Snapshot hyperparameters/config for this run, alongside auto-captured
        metadata (SLURM job ID, git commit, start time). Call this once, right
        after setup_run(), with whatever dict of hyperparameters you're using."""
        payload = {
            "run_id": self.run_id,
            "slurm_job_id": _get_slurm_job_id(),
            "git_commit": _get_git_commit(),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "hostname": os.environ.get("SLURMD_NODENAME", os.uname().nodename),
            "config": config,
        }
        with open(self.root / "run_config.json", "w") as f:
            json.dump(payload, f, indent=2)

    def checkpoint_path(self, filename: str) -> Path:
        """Convenience: self.checkpoints / filename, as a Path."""
        return self.checkpoints / filename

    def result_path(self, filename: str) -> Path:
        return self.results / filename

    def figure_path(self, filename: str) -> Path:
        return self.figures / filename


def _install_exception_hook(logger: logging.Logger) -> None:
    """Makes sure that ANY uncaught exception (the kind that currently just
    prints a traceback to stderr and vanishes into a .err file) also gets
    written, with full traceback, into run.log inside the run folder itself.
    This means the error lives right next to the checkpoints/config it was
    produced with, instead of only in a SLURM .err file you have to go hunt
    down separately by job ID."""
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error(
            "Uncaught exception — run failed:\n%s",
            "".join(traceback.format_exception(exc_type, exc_value, exc_traceback)),
        )
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = handle_exception


def _setup_logging(run: RunPaths) -> logging.Logger:
    logger = logging.getLogger(run.run_id)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        # Already configured (e.g. setup_run called twice) — don't duplicate handlers
        return logger

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(run.logs / "run.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    _install_exception_hook(logger)
    return logger


def setup_run(model_name: str, base_dir: str, extra_config: dict = None) -> RunPaths:
    """Creates a brand-new run: unique run_id, full directory tree on scratch,
    and logging (console + file) with automatic uncaught-exception capture.

    Call this ONCE at the very top of main_cv.py / evaluate.py, before doing
    anything else. Returns a RunPaths object — pass it around (or its
    sub-paths) anywhere you currently reference CHECKPOINT_DIR / OUTPUT_DIR.
    """
    run_id = _make_run_id(model_name)
    root = Path(base_dir) / run_id

    paths = RunPaths(
        run_id=run_id,
        root=root,
        checkpoints=root / "checkpoints",
        figures=root / "figures",
        results=root / "results",
        logs=root / "logs",
    )
    for p in (paths.root, paths.checkpoints, paths.figures, paths.results, paths.logs):
        p.mkdir(parents=True, exist_ok=True)

    logger = _setup_logging(paths)
    logger.info("Run started: %s", run_id)
    logger.info("Output root: %s", root)

    if extra_config is not None:
        paths.save_config(extra_config)

    return paths


def load_run(run_id: str, base_dir: str) -> RunPaths:
    """Points at an EXISTING run folder rather than creating a new one — use
    this in evaluate.py when you want to run evaluation against a specific
    past training run's checkpoints, instead of the most recent run.

    Raises FileNotFoundError with a helpful message (and a list of what IS
    available) if the run_id doesn't exist, rather than silently creating
    an empty folder and confusingly finding no checkpoints later.
    """
    root = Path(base_dir) / run_id
    if not root.exists():
        available = sorted(p.name for p in Path(base_dir).glob("*") if p.is_dir())
        raise FileNotFoundError(
            f"No run found at {root}.\nAvailable runs in {base_dir}:\n  "
            + "\n  ".join(available[-15:])  # last 15, most recent runs sort last
        )
    paths = RunPaths(
        run_id=run_id,
        root=root,
        checkpoints=root / "checkpoints",
        figures=root / "figures",
        results=root / "results",
        logs=root / "logs",
    )
    _setup_logging(paths)
    return paths


def list_runs(base_dir: str, model_name: str = None) -> list:
    """Returns run_ids under base_dir, optionally filtered by model name
    prefix, sorted with most recent last. Handy for a quick
    `python -c "from run_manager import list_runs; ..."` check, or for
    building a --run_id latest convenience flag in your CLI."""
    base = Path(base_dir)
    if not base.exists():
        return []
    runs = sorted(p.name for p in base.glob("*") if p.is_dir())
    if model_name:
        runs = [r for r in runs if r.startswith(model_name + "_")]
    return runs


def latest_run(base_dir: str, model_name: str) -> str:
    """Returns the most recent run_id for a given model, or raises
    FileNotFoundError if none exist yet. Useful for a --run_id latest flag."""
    runs = list_runs(base_dir, model_name=model_name)
    if not runs:
        raise FileNotFoundError(f"No runs found for model '{model_name}' in {base_dir}")
    base = Path(base_dir)
    return max(runs, key=lambda run_id: (base / run_id).stat().st_mtime)
