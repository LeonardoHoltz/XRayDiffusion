# Training Hyperparameters
INPUT_SIZE = 65536 # 784 for mnist
NUM_CLASSES = 2
LEARNING_RATE = 0.0001
NUM_EPOCHS = 50
N_FEATURES = 128 # 128 ok, 256 better (but slower)
BETAS = (1e-4, 0.02)
N_T = 500

# Dataset
DATA_DIR = "AiresPucrs/chest-xray"
NUM_WORKERS = 1
BATCH_SIZE = 1

# Compute related
ACCELERATOR = "cpu"
DEVICES = 1
PRECISION = 32
MODEL_CHECKPOINT_DIR = "./checkpoints/"