# config.py
import os

# config.py
DEVICE = 'cpu'  #remove. this is for test running

# Paths
BASE_PATH = "/home/shadow/scratch/MRI_DATASET/data"
OUTPUT_DIR = "./outputs"
CHECKPOINT_DIR = "./checkpoints"  

# Scanner parameters
N_PE = 136
N_RO = 150
N_SLICES = 38

# Data split subjects
TRAIN_SUBJECTS = ['9092', '9133', '9147', '9139', '9110']
VAL_SUBJECTS = ['9101', '9074']
TEST_SUBJECTS = ['9033', '9070']
CV_SUBJECTS = ['9033', '9070', '9074', '9092', '9101', '9110', '9133', '9139', '9147']
TEST_SUBJECTS_CV = ['9033', '9070']  # Held out permanently

# Training defaults
DEFAULT_ACCELERATION = 2
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 8
DEFAULT_SEED = 42

# Create directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
