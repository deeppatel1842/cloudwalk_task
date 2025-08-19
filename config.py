import torch
import os

# Project Level Configuration
PROJECT_NAME = "Cloudwalk_Spoken_Digit_Classification"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# Data Configuration
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_URL = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip"
RECORDINGS_PATH = os.path.join(DATA_DIR, "free-spoken-digit-dataset-master", "recordings")

# Audio Feature Extraction Configuration
SAMPLE_RATE = 8000
N_FFT = 512
HOP_LENGTH = 256
N_MELS = 64
DURATION = 2.0  # seconds
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

# Training Configuration
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 30 # Increased epochs for better convergence
WEIGHT_DECAY = 1e-5
TEST_SPLIT_SIZE = 0.1
VALIDATION_SPLIT_SIZE = 0.1
SCHEDULER_STEP_SIZE = 10 # For learning rate scheduler
SCHEDULER_GAMMA = 0.1    # For learning rate scheduler

# Model Configurations
CNN_CONFIG = {
    "in_channels": 1,
    "num_classes": 10,
    "model_name": "Improved_CNN_MFCC" 
}

RNN_CONFIG = {
    "input_size": N_MELS,
    "hidden_size": 128,
    "num_layers": 3,
    "num_classes": 10,
    "model_name": "Bidirectional_LSTM_MFCC" 
}

# Paths
SAVED_MODELS_PATH = os.path.join(BASE_DIR, "saved_models")
RESULTS_PATH = os.path.join(BASE_DIR, "results")
LOGS_PATH = os.path.join(BASE_DIR, "logs")

# Microphone Inference Configuration
MIC_SAMPLE_RATE = 8000
MIC_CHUNK_SIZE = 1024
