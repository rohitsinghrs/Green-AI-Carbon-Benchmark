import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "adult"
OUTPUT_DIR = BASE_DIR / "prepared_data"

TRAIN_FILE = DATA_DIR / "adult.data"
TEST_FILE = DATA_DIR / "adult.test"

OUTPUT_DIR.mkdir(exist_ok=True)

# Adult dataset column names
COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income"
]

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING ADULT DATASET")
print("=" * 60)

train_df = pd.read_csv(
    TRAIN_FILE,
    names=COLUMNS,
    sep=",",
    skipinitialspace=True,
    na_values="?"
)

test_df = pd.read_csv(
    TEST_FILE,
    names=COLUMNS,
    sep=",",
    skipinitialspace=True,
    na_values="?",
    skiprows=1
)

print("\nOriginal training shape:", train_df.shape)
print("Original testing shape :", test_df.shape)

# ============================================================
# CLEAN STRING VALUES
# ============================================================

for df in [train_df, test_df]:

    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].str.strip()

# adult.test labels contain a period:
# ">50K." and "<=50K."
test_df["income"] = test_df["income"].str.replace(".", "", regex=False)

# ============================================================
# REMOVE ROWS WITH MISSING VALUES
# ============================================================

print("\nMissing values before cleaning:")
print(train_df.isnull().sum())

print("\nRemoving rows containing missing values...")

train_df = train_df.dropna().reset_index(drop=True)
test_df = test_df.dropna().reset_index(drop=True)

print("\nTraining shape after cleaning:", train_df.shape)
print("Testing shape after cleaning :", test_df.shape)

# ============================================================
# CONVERT TARGET TO BINARY
# ============================================================

train_df["income"] = train_df["income"].map({
    "<=50K": 0,
    ">50K": 1
})

test_df["income"] = test_df["income"].map({
    "<=50K": 0,
    ">50K": 1
})

# Remove any unexpected target values
train_df = train_df.dropna(subset=["income"])
test_df = test_df.dropna(subset=["income"])

train_df["income"] = train_df["income"].astype(int)
test_df["income"] = test_df["income"].astype(int)

# ============================================================
# SAVE CLEAN DATA
# ============================================================

train_output = OUTPUT_DIR / "adult_train_clean.csv"
test_output = OUTPUT_DIR / "adult_test_clean.csv"

train_df.to_csv(train_output, index=False)
test_df.to_csv(test_output, index=False)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETE")
print("=" * 60)

print("\nFinal training samples:", len(train_df))
print("Final testing samples :", len(test_df))

print("\nTraining class distribution:")
print(train_df["income"].value_counts())

print("\nTesting class distribution:")
print(test_df["income"].value_counts())

print("\nSaved files:")
print(train_output)
print(test_output)

print("\n" + "=" * 60)