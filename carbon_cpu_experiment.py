import time
import random
import os

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

from codecarbon import EmissionsTracker

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

HARDWARE_NAME = "AMD_CPU"

RESULTS_FILE = "cpu_carbon_results.csv"

CODECARBON_DIR = "codecarbon_cpu"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


# ============================================================
# LOAD + PREPROCESS DATA
# ============================================================

def load_data():

    train_df = pd.read_csv(TRAIN_FILE)

    test_df = pd.read_csv(TEST_FILE)

    X_train = train_df.drop(
        columns=[TARGET]
    )

    y_train = train_df[TARGET].values

    X_test = test_df.drop(
        columns=[TARGET]
    )

    y_test = test_df[TARGET].values

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

    X_train = preprocessor.fit_transform(
        X_train
    )

    X_test = preprocessor.transform(
        X_test
    )

    X_train = np.asarray(
        X_train,
        dtype=np.float32
    )

    X_test = np.asarray(
        X_test,
        dtype=np.float32
    )

    y_train = np.asarray(
        y_train,
        dtype=np.float32
    )

    y_test = np.asarray(
        y_test,
        dtype=np.float32
    )

    return (
        X_train,
        y_train,
        X_test,
        y_test
    )


# ============================================================
# TRAIN
# ============================================================

def train_model(
    model,
    train_loader
):

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

            X_batch = X_batch.to(
                DEVICE
            )

            y_batch = y_batch.to(
                DEVICE
            )

            optimizer.zero_grad()

            outputs = model(
                X_batch
            )

            loss = criterion(
                outputs,
                y_batch
            )

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

    end_time = time.perf_counter()

    return (
        end_time - start_time
    )


# ============================================================
# INFERENCE + METRICS
# ============================================================

def evaluate_model(
    model,
    test_loader
):

    model.eval()

    predictions = []

    actual = []

    start_time = time.perf_counter()

    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            X_batch = X_batch.to(
                DEVICE
            )

            outputs = model(
                X_batch
            )

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

    inference_time = (
        end_time - start_time
    )

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
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 70)
    print("STEP 9 - CPU + CODECARBON EXPERIMENT")
    print("=" * 70)

    os.makedirs(
        CODECARBON_DIR,
        exist_ok=True
    )

    (
        X_train,
        y_train,
        X_test,
        y_test
    ) = load_data()

    input_dim = X_train.shape[1]

    print("\nHardware:", HARDWARE_NAME)

    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
    )

    print(
        "Input features:",
        input_dim
    )

    print(
        "Epochs:",
        EPOCHS
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    results = []

    for model_name in MODELS:

        print("\n" + "=" * 70)

        print(
            "MODEL:",
            model_name
        )

        print("=" * 70)

        # ----------------------------------------------------
        # Reset seed
        # ----------------------------------------------------

        set_seed(SEED)

        # ----------------------------------------------------
        # Model creation
        # ----------------------------------------------------

        model = create_model(
            model_name,
            input_dim
        )

        model.to(DEVICE)

        parameter_count = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        print(
            "Parameters:",
            parameter_count
        )

        # ----------------------------------------------------
        # DataLoaders
        # ----------------------------------------------------

        train_dataset = TensorDataset(
            torch.tensor(X_train),
            torch.tensor(y_train)
        )

        test_dataset = TensorDataset(
            torch.tensor(X_test),
            torch.tensor(y_test)
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

        # ----------------------------------------------------
        # CODECARBON START
        # ----------------------------------------------------

        tracker = EmissionsTracker(
            project_name=f"adult_{model_name}_cpu",
            output_dir=CODECARBON_DIR,
            log_level="error"
        )

        print("\nStarting CodeCarbon...")

        tracker.start()

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        training_time = train_model(
            model,
            train_loader
        )

        # ----------------------------------------------------
        # CODECARBON STOP
        # ----------------------------------------------------

        emissions_kg = tracker.stop()

        # ----------------------------------------------------
        # INFERENCE
        # ----------------------------------------------------

        (
            accuracy,
            precision,
            recall,
            f1,
            inference_time
        ) = evaluate_model(
            model,
            test_loader
        )

        emissions_g = (
            emissions_kg * 1000
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        result = {

            "architecture":
                model_name,

            "hardware":
                HARDWARE_NAME,

            "repeat":
                1,

            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "training_time_sec":
                training_time,

            "inference_time_sec":
                inference_time,

            "parameters":
                parameter_count,

            "emissions_kg":
                emissions_kg,

            "emissions_g":
                emissions_g
        }

        results.append(
            result
        )

        print("\nRESULT")

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
            f"F1             : {f1:.4f}"
        )

        print(
            f"Training time  : "
            f"{training_time:.4f} sec"
        )

        print(
            f"Inference time : "
            f"{inference_time:.4f} sec"
        )

        print(
            f"CO2            : "
            f"{emissions_g:.8f} g"
        )

    # ========================================================
    # SAVE
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    print("\n" + "=" * 70)

    print(
        "CPU + CODECARBON EXPERIMENT COMPLETE"
    )

    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        "\nSaved:",
        RESULTS_FILE
    )

    print(
        "CodeCarbon directory:",
        CODECARBON_DIR
    )

    print("=" * 70)


if __name__ == "__main__":
    main()