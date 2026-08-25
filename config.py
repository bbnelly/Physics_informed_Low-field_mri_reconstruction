# config.py
import os

# config.py
import os

# Paths
BASE_PATH = os.environ.get(
    "MRI_DATA_PATH",
    os.path.expanduser("~/scratch/MRI_DATASET/data"),
)
OUTPUT_DIR = "./outputs"
CHECKPOINT_DIR = "./checkpoints"

# Scanner parameters
N_PE = 136
N_RO = 150
N_SLICES = 38

# Data split subjects
CV_SUBJECTS = ['9033', '9070', '9074', '9092', '9101', '9110', '9133', '9139', '9147']

# Training defaults
DEFAULT_ACCELERATION = 2
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_SEED = 42

# Create directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
