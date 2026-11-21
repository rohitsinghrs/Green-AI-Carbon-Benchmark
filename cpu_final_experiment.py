import os
import time
import random
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from codecarbon import EmissionsTracker

from pipeline import (
    X_train_processed,
    X_test_processed,
    y_train_np,
    y_test_np,
    DEVICE,
    SEED
)

from models import create_model


# ============================================================
# CONFIGURATION
# ============================================================

EPOCHS = 20
BATCH_SIZE = 256

HARDWARE = "AMD_CPU"

MODELS = [
    "Linear",
    "SmallMLP",
    "DeepMLP",
    "CNN1D"
]

RESULT_FILE = "cpu_final_results.csv"
CODECARBON_DIR = "codecarbon_final"


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("FINAL CPU + CODECARBON EXPERIMENT")
print("=" * 70)

print(f"\nHardware        : {HARDWARE}")
print(f"Training samples: {len(X_train_processed)}")
print(f"Testing samples : {len(X_test_processed)}")
print(f"Input features  : {X_train_processed.shape[1]}")
print(f"Epochs          : {EPOCHS}")
print(f"Batch size      : {BATCH_SIZE}")

print("\nModels:")
for model_name in MODELS:
    print(" -", model_name)


# ============================================================
# DATA TO TORCH
# ============================================================

X_train = torch.tensor(
    X_train_processed,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test_processed,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train_np,
    dtype=torch.float32
)

y_test = torch.tensor(
    y_test_np,
    dtype=torch.float32
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(model):

    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    criterion = torch.nn.BCEWithLogitsLoss()

    n_samples = len(X_train)

    start_time = time.perf_counter()

    for epoch in range(EPOCHS):

        # Deterministic shuffle based on torch random state
        permutation = torch.randperm(n_samples)

        for i in range(0, n_samples, BATCH_SIZE):

            indices = permutation[i:i + BATCH_SIZE]

            batch_x = X_train[indices]
            batch_y = y_train[indices]

            optimizer.zero_grad()

            outputs = model(batch_x)

            loss = criterion(
                outputs,
                batch_y
            )

            loss.backward()

            optimizer.step()

    training_time = time.perf_counter() - start_time

    return training_time


# ============================================================
# INFERENCE FUNCTION
# ============================================================

def evaluate_model(model):

    model.eval()

    start_time = time.perf_counter()

    with torch.no_grad():

        outputs = model(X_test)

        probabilities = torch.sigmoid(outputs)

        predictions = (
            probabilities >= 0.5
        ).float()

    inference_time = time.perf_counter() - start_time

    y_true = y_test.numpy()
    y_pred = predictions.numpy()

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
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

os.makedirs(
    CODECARBON_DIR,
    exist_ok=True
)


for model_name in MODELS:

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Reset seed for comparable initialization
    # --------------------------------------------------------

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_model(
        model_name,
        X_train_processed.shape[1]
    )

    model = model.to(DEVICE)

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Parameters:", parameters)

    # --------------------------------------------------------
    # CodeCarbon tracker
    # --------------------------------------------------------

    project_name = (
        f"adult_{model_name.lower()}_cpu_final"
    )

    tracker = EmissionsTracker(
        project_name=project_name,
        output_dir=CODECARBON_DIR,
        measure_power_secs=1,
        log_level="error"
    )

    print("Starting CodeCarbon...")

    tracker.start()

    try:

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        training_time = train_model(model)

        # ----------------------------------------------------
        # INFERENCE
        # ----------------------------------------------------

        (
            accuracy,
            precision,
            recall,
            f1,
            inference_time
        ) = evaluate_model(model)

    finally:

        emissions_kg = tracker.stop()


    # --------------------------------------------------------
    # Energy information
    # --------------------------------------------------------

    emissions_g = (
        emissions_kg * 1000
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

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
        f"Training time  : {training_time:.4f} sec"
    )

    print(
        f"Inference time : {inference_time:.4f} sec"
    )

    print(
        f"CO2            : {emissions_g:.6f} g"
    )


    results.append({

        "architecture": model_name,

        "hardware": HARDWARE,

        "repeat": 1,

        "seed": SEED,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "training_time_sec":
            training_time,

        "inference_time_sec":
            inference_time,

        "parameters":
            parameters,

        "emissions_kg":
            emissions_kg,

        "emissions_g":
            emissions_g

    })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)


print("\n" + "=" * 70)
print("FINAL CPU EXPERIMENT COMPLETE")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


results_df.to_csv(
    RESULT_FILE,
    index=False
)


print("\nSaved:")
print(f"- {RESULT_FILE}")
print(f"- {CODECARBON_DIR}/")

print("=" * 70)