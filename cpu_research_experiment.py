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
    DEVICE
)

from models import create_model


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

EPOCHS = 20
BATCH_SIZE = 256
REPETITIONS = 3

SEEDS = [42, 43, 44]

HARDWARE = "AMD_CPU"

MODELS = [
    "Linear",
    "SmallMLP",
    "DeepMLP",
    "CNN1D"
]

OUTPUT_DIR = "codecarbon_research_cpu"
RAW_FILE = "cpu_research_raw.csv"
SUMMARY_FILE = "cpu_research_summary.csv"


# ============================================================
# DATA
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
# TRAINING
# ============================================================

def train_model(model):

    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    criterion = torch.nn.BCEWithLogitsLoss()

    n_samples = len(X_train)

    start = time.perf_counter()

    for epoch in range(EPOCHS):

        permutation = torch.randperm(
            n_samples
        )

        for i in range(
            0,
            n_samples,
            BATCH_SIZE
        ):

            indices = permutation[
                i:i + BATCH_SIZE
            ]

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

    return time.perf_counter() - start


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model):

    model.eval()

    start = time.perf_counter()

    with torch.no_grad():

        outputs = model(X_test)

        probabilities = torch.sigmoid(
            outputs
        )

        predictions = (
            probabilities >= 0.5
        ).float()

    inference_time = (
        time.perf_counter() - start
    )

    true = y_test.numpy()
    pred = predictions.numpy()

    return {
        "accuracy": accuracy_score(
            true,
            pred
        ),

        "precision": precision_score(
            true,
            pred,
            zero_division=0
        ),

        "recall": recall_score(
            true,
            pred,
            zero_division=0
        ),

        "f1": f1_score(
            true,
            pred,
            zero_division=0
        ),

        "inference_time_sec":
            inference_time
    }


# ============================================================
# EXPERIMENT
# ============================================================

print("=" * 70)
print("STEP 12 - RESEARCH-GRADE CPU EXPERIMENT")
print("=" * 70)

print()
print("Hardware        :", HARDWARE)
print("Train samples   :", len(X_train))
print("Test samples    :", len(X_test))
print("Features        :", X_train.shape[1])
print("Epochs          :", EPOCHS)
print("Batch size      :", BATCH_SIZE)
print("Repetitions     :", REPETITIONS)
print("Seeds           :", SEEDS)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

results = []


for model_name in MODELS:

    for repetition, seed in enumerate(
        SEEDS,
        start=1
    ):

        print()
        print("=" * 70)
        print(
            f"MODEL: {model_name} | "
            f"REPETITION: {repetition}/{REPETITIONS} | "
            f"SEED: {seed}"
        )
        print("=" * 70)

        # ----------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        model = create_model(
            model_name,
            X_train.shape[1]
        )

        model = model.to(DEVICE)

        parameters = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        print(
            "Parameters:",
            parameters
        )

        # ----------------------------------------------------
        # CodeCarbon
        # ----------------------------------------------------

        project_name = (
            f"adult_{model_name}"
            f"_cpu_rep_{repetition}"
            f"_seed_{seed}"
        )

        tracker = EmissionsTracker(
            project_name=project_name,
            output_dir=OUTPUT_DIR,
            measure_power_secs=1,
            log_level="error"
        )

        print("Starting CodeCarbon...")

        tracker.start()

        try:

            training_time = train_model(
                model
            )

            metrics = evaluate_model(
                model
            )

        finally:

            emissions_kg = tracker.stop()

        emissions_g = (
            emissions_kg * 1000
        )

        print()
        print("RESULT")

        print(
            f"Accuracy       : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Precision      : "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall         : "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"F1             : "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"Training time  : "
            f"{training_time:.4f} sec"
        )

        print(
            f"Inference time : "
            f"{metrics['inference_time_sec']:.4f} sec"
        )

        print(
            f"CO2            : "
            f"{emissions_g:.6f} g"
        )

        results.append({

            "architecture":
                model_name,

            "hardware":
                HARDWARE,

            "repeat":
                repetition,

            "seed":
                seed,

            "accuracy":
                metrics["accuracy"],

            "precision":
                metrics["precision"],

            "recall":
                metrics["recall"],

            "f1":
                metrics["f1"],

            "training_time_sec":
                training_time,

            "inference_time_sec":
                metrics[
                    "inference_time_sec"
                ],

            "parameters":
                parameters,

            "emissions_kg":
                emissions_kg,

            "emissions_g":
                emissions_g
        })


# ============================================================
# RAW RESULTS
# ============================================================

raw_df = pd.DataFrame(
    results
)

raw_df.to_csv(
    RAW_FILE,
    index=False
)


# ============================================================
# SUMMARY: MEAN ± STD
# ============================================================

summary = (
    raw_df
    .groupby(
        ["architecture", "hardware"]
    )
    .agg(
        accuracy_mean=(
            "accuracy",
            "mean"
        ),

        accuracy_std=(
            "accuracy",
            "std"
        ),

        precision_mean=(
            "precision",
            "mean"
        ),

        precision_std=(
            "precision",
            "std"
        ),

        recall_mean=(
            "recall",
            "mean"
        ),

        recall_std=(
            "recall",
            "std"
        ),

        f1_mean=(
            "f1",
            "mean"
        ),

        f1_std=(
            "f1",
            "std"
        ),

        training_time_mean=(
            "training_time_sec",
            "mean"
        ),

        training_time_std=(
            "training_time_sec",
            "std"
        ),

        inference_time_mean=(
            "inference_time_sec",
            "mean"
        ),

        inference_time_std=(
            "inference_time_sec",
            "std"
        ),

        emissions_mean_g=(
            "emissions_g",
            "mean"
        ),

        emissions_std_g=(
            "emissions_g",
            "std"
        )
    )
    .reset_index()
)


summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 70)
print("RESEARCH-GRADE CPU EXPERIMENT COMPLETE")
print("=" * 70)

print()
print("RAW RESULTS")
print(
    raw_df.to_string(
        index=False
    )
)

print()
print("=" * 70)
print("SUMMARY: MEAN ± STD")
print("=" * 70)

print(
    summary.to_string(
        index=False
    )
)

print()
print("Saved files:")
print("-", RAW_FILE)
print("-", SUMMARY_FILE)
print("-", OUTPUT_DIR + "/")

print("=" * 70)