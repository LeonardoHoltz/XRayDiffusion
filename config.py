# Training Hyperparameters
INPUT_SIZE = 65536 # 784 for mnist
NUM_CLASSES = 2
LEARNING_RATE = 0.0001
NUM_EPOCHS = 25
N_FEATURES = 128 # 128 ok, 256 better (but slower)
BETAS = (1e-4, 0.02)
N_T = 500

# Dataset
DATA_DIR = "AiresPucrs/chest-xray"
NUM_WORKERS = 2
BATCH_SIZE = 8

# Compute related
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = 32

MODEL_CHECKPOINT_DIR = "./checkpoints/"