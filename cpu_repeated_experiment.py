import time
import random
import os
import glob

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

REPEATS = 3

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

RAW_RESULTS_FILE = "cpu_repeated_results.csv"
SUMMARY_RESULTS_FILE = "cpu_summary.csv"

CODECARBON_DIR = "codecarbon_repeated"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# LOAD + PREPROCESS
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
# TRAINING
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

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

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

    end_time = time.perf_counter()

    return end_time - start_time


# ============================================================
# EVALUATION
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

            X_batch = X_batch.to(DEVICE)

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
# GET LATEST CODECARBON RECORD
# ============================================================

def get_latest_carbon_record(
    before_files
):

    after_files = set(
        glob.glob(
            os.path.join(
                CODECARBON_DIR,
                "emissions.csv"
            )
        )
    )

    # CodeCarbon normally writes to one emissions.csv.
    # Read the latest record from it.

    emissions_file = os.path.join(
        CODECARBON_DIR,
        "emissions.csv"
    )

    if not os.path.exists(
        emissions_file
    ):
        raise FileNotFoundError(
            "CodeCarbon emissions.csv was not found."
        )

    df = pd.read_csv(
        emissions_file
    )

    if len(df) == 0:
        raise RuntimeError(
            "CodeCarbon emissions.csv is empty."
        )

    return df.iloc[-1].to_dict()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("STEP 10 - REPEATED CPU + CODECARBON EXPERIMENT")
    print("=" * 75)

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
    print("Training samples:", len(X_train))
    print("Testing samples :", len(X_test))
    print("Input features   :", input_dim)
    print("Epochs           :", EPOCHS)
    print("Batch size       :", BATCH_SIZE)
    print("Repetitions      :", REPEATS)

    all_results = []

    # ========================================================
    # EXPERIMENT LOOP
    # ========================================================

    for model_name in MODELS:

        for repeat in range(
            1,
            REPEATS + 1
        ):

            print("\n" + "=" * 75)

            print(
                f"MODEL: {model_name} | "
                f"REPETITION: {repeat}/{REPEATS}"
            )

            print("=" * 75)

            # ------------------------------------------------
            # Reproducibility
            # ------------------------------------------------

            run_seed = (
                SEED + repeat - 1
            )

            set_seed(
                run_seed
            )

            # ------------------------------------------------
            # Model
            # ------------------------------------------------

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

            # ------------------------------------------------
            # DataLoader
            # ------------------------------------------------

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

            # ------------------------------------------------
            # CodeCarbon
            # ------------------------------------------------

            tracker = EmissionsTracker(
                project_name=(
                    f"adult_{model_name}"
                    f"_cpu_repeat_{repeat}"
                ),
                output_dir=CODECARBON_DIR,
                log_level="error"
            )

            print(
                "Starting CodeCarbon..."
            )

            tracker.start()

            # ------------------------------------------------
            # Training
            # ------------------------------------------------

            training_time = train_model(
                model,
                train_loader
            )

            # ------------------------------------------------
            # Stop Carbon Tracking
            # ------------------------------------------------

            emissions_kg = tracker.stop()

            # ------------------------------------------------
            # Evaluation
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Read CodeCarbon
            # ------------------------------------------------

            carbon = get_latest_carbon_record(
                None
            )

            energy_kwh = float(
                carbon[
                    "energy_consumed"
                ]
            )

            emissions_kg_csv = float(
                carbon[
                    "emissions"
                ]
            )

            carbon_duration = float(
                carbon[
                    "duration"
                ]
            )

            cpu_power = float(
                carbon[
                    "cpu_power"
                ]
            )

            ram_power = float(
                carbon[
                    "ram_power"
                ]
            )

            cpu_energy = float(
                carbon[
                    "cpu_energy"
                ]
            )

            ram_energy = float(
                carbon[
                    "ram_energy"
                ]
            )

            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            result = {

                "architecture":
                    model_name,

                "hardware":
                    HARDWARE_NAME,

                "repeat":
                    repeat,

                "seed":
                    run_seed,

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

                "codecarbon_duration_sec":
                    carbon_duration,

                "energy_kwh":
                    energy_kwh,

                "cpu_energy_kwh":
                    cpu_energy,

                "ram_energy_kwh":
                    ram_energy,

                "emissions_kg":
                    emissions_kg_csv,

                "emissions_g":
                    emissions_kg_csv * 1000,

                "cpu_power_w":
                    cpu_power,

                "ram_power_w":
                    ram_power
            }

            all_results.append(
                result
            )

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            print("\nRESULT")

            print(
                f"Accuracy      : {accuracy:.4f}"
            )

            print(
                f"F1            : {f1:.4f}"
            )

            print(
                f"Training time : "
                f"{training_time:.4f} sec"
            )

            print(
                f"Energy        : "
                f"{energy_kwh:.10f} kWh"
            )

            print(
                f"CO2           : "
                f"{emissions_kg_csv * 1000:.6f} g"
            )

    # ========================================================
    # RAW RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_df.to_csv(
        RAW_RESULTS_FILE,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = (
        results_df
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

            energy_mean_kwh=(
                "energy_kwh",
                "mean"
            ),

            energy_std_kwh=(
                "energy_kwh",
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
        SUMMARY_RESULTS_FILE,
        index=False
    )

    # ========================================================
    # FINAL DISPLAY
    # ========================================================

    print("\n" + "=" * 75)
    print("REPEATED CPU EXPERIMENT COMPLETE")
    print("=" * 75)

    print("\nRAW RESULTS")
    print(
        results_df.to_string(
            index=False
        )
    )

    print("\nSUMMARY: MEAN ± STD")

    print(
        summary.to_string(
            index=False
        )
    )

    print("\nSaved files:")

    print(
        "-",
        RAW_RESULTS_FILE
    )

    print(
        "-",
        SUMMARY_RESULTS_FILE
    )

    print(
        "-",
        CODECARBON_DIR
    )

    print("=" * 75)


if __name__ == "__main__":
    main()