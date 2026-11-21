import random
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 60)
print("STEP 5 - PREPROCESSING PIPELINE")
print("=" * 60)

# ============================================================
# 2. LOAD FIXED DATASET
# ============================================================

TRAIN_FILE = "prepared_data/adult_train_clean.csv"
TEST_FILE = "prepared_data/adult_test_clean.csv"

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("\nTraining shape:", train_df.shape)
print("Testing shape :", test_df.shape)

# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

TARGET = "income"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

# ============================================================
# 4. IDENTIFY COLUMN TYPES
# ============================================================

categorical_columns = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_columns = X_train.select_dtypes(
    include=[np.number]
).columns.tolist()

print("\nNumerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)

# ============================================================
# 5. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_columns
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_columns
        )
    ]
)

# IMPORTANT:
# Fit ONLY on training data.
# Test data is only transformed.

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed training shape:", X_train_processed.shape)
print("Processed testing shape :", X_test_processed.shape)

# ============================================================
# 6. CONVERT TO NUMPY
# ============================================================

X_train_processed = np.asarray(
    X_train_processed,
    dtype=np.float32
)

X_test_processed = np.asarray(
    X_test_processed,
    dtype=np.float32
)

y_train_np = y_train.to_numpy(dtype=np.float32)
y_test_np = y_test.to_numpy(dtype=np.float32)

# ============================================================
# 7. HARDWARE
# ============================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("\nHardware:", torch.cuda.get_device_name(0))
else:
    DEVICE = torch.device("cpu")
    print("\nHardware: CPU")

# ============================================================
# 8. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PIPELINE READY")
print("=" * 60)

print("Random seed:", SEED)
print("Device:", DEVICE)
print("Train samples:", len(X_train_processed))
print("Test samples:", len(X_test_processed))
print("Input features:", X_train_processed.shape[1])

print("=" * 60)