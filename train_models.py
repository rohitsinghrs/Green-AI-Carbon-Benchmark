import time
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from models import create_model


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 0.001

TRAIN_FILE = "prepared_data/adult_train_clean.csv"
TEST_FILE = "prepared_data/adult_test_clean.csv"

TARGET = "income"

DEVICE = torch.device("cpu")

MODELS = [
    "Linear",
    "SmallMLP",
    "DeepMLP",
    "CNN1D"
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 7 - MODEL TRAINING")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)


X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET].values

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET].values


# ============================================================
# PREPROCESSING
# ============================================================

categorical_columns = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_columns = X_train.select_dtypes(
    include=[np.number]
).columns.tolist()


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


X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


X_train = np.asarray(X_train, dtype=np.float32)
X_test = np.asarray(X_test, dtype=np.float32)

y_train = np.asarray(y_train, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.float32)


INPUT_DIM = X_train.shape[1]


print("\nDataset")
print("Train samples:", len(X_train))
print("Test samples :", len(X_test))
print("Input features:", INPUT_DIM)


# ============================================================
# PYTORCH DATASETS
# ============================================================

X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)

X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test)


train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(model):

    model.to(DEVICE)

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    model.train()

    start_time = time.perf_counter()

    for epoch in range(EPOCHS):

        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(X_batch)

            loss = criterion(
                outputs,
                y_batch
            )

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 5 == 0:

            average_loss = (
                epoch_loss / len(train_loader)
            )

            print(
                f"    Epoch {epoch + 1:02d}/{EPOCHS} "
                f"- Loss: {average_loss:.4f}"
            )

    end_time = time.perf_counter()

    training_time = end_time - start_time

    return training_time


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model):

    model.eval()

    predictions = []

    actual = []

    start_time = time.perf_counter()

    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            X_batch = X_batch.to(DEVICE)

            outputs = model(X_batch)

            probabilities = torch.sigmoid(
                outputs
            )

            batch_predictions = (
                probabilities >= 0.5
            ).int().cpu().numpy()

            predictions.extend(
                batch_predictions
            )

            actual.extend(
                y_batch.numpy()
            )

    end_time = time.perf_counter()

    inference_time = end_time - start_time

    accuracy = accuracy_score(
        actual,
        predictions
    )

    precision = precision_score(
        actual,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        actual,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        actual,
        predictions,
        zero_division=0
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        inference_time
    )


# ============================================================
# RUN EXPERIMENT
# ============================================================

results = []


for model_name in MODELS:

    print("\n" + "=" * 70)

    print(
        f"Training model: {model_name}"
    )

    print("=" * 70)

    # Reset seed before every model
    torch.manual_seed(SEED)

    model = create_model(
        model_name,
        INPUT_DIM
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Trainable parameters:",
        parameter_count
    )

    # -------------------------------
    # TRAIN
    # -------------------------------

    training_time = train_model(model)

    # -------------------------------
    # EVALUATE
    # -------------------------------

    (
        accuracy,
        precision,
        recall,
        f1,
        inference_time
    ) = evaluate_model(model)

    print("\nResults:")

    print(
        f"Accuracy       : {accuracy:.4f}"
    )

    print(
        f"Precision      : {precision:.4f}"
    )

    print(
        f"Recall         : {recall:.4f}"
    )

    print(
        f"F1 Score       : {f1:.4f}"
    )

    print(
        f"Training time  : {training_time:.4f} sec"
    )

    print(
        f"Inference time : {inference_time:.4f} sec"
    )

    results.append({

        "architecture": model_name,

        "hardware": "AMD_CPU",

        "repeat": 1,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "training_time_sec": training_time,

        "inference_time_sec": inference_time,

        "parameters": parameter_count

    })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "cpu_performance_results.csv",
    index=False
)


print("\n" + "=" * 70)
print("CPU PERFORMANCE EXPERIMENT COMPLETE")
print("=" * 70)

print(results_df.to_string(index=False))

print("\nSaved:")
print("cpu_performance_results.csv")

print("=" * 70)