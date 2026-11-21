# ============================================================
# GREEN AI HARDWARE EXPERIMENT CONFIGURATION
# ============================================================

RANDOM_SEED = 42

# Dataset
TRAIN_FILE = "prepared_data/adult_train_clean.csv"
TEST_FILE = "prepared_data/adult_test_clean.csv"

# Target
TARGET_COLUMN = "income"

# Experimental repetitions
N_REPEATS = 3

# Training
EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 0.001

# Hardware labels
LOCAL_HARDWARE = "AMD_CPU"
GPU_HARDWARE = "COLAB_GPU"

# Models
MODELS = [
    "Logistic_Regression",
    "Small_MLP",
    "Deep_MLP",
    "1D_CNN"
]

print("=" * 60)
print("GREEN AI EXPERIMENT CONFIGURATION")
print("=" * 60)

print("Random seed:", RANDOM_SEED)
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)
print("Repeats:", N_REPEATS)

print("\nModels:")
for model in MODELS:
    print("-", model)