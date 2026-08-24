# Project 5: Student GPA Prediction using Linear Regression
#
# Dataset:
#   student GPA data bank.xlsx
#
# The Excel dataset should be kept in the same folder as this Python file.

import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# ============================================================
# Step 0: User-editable variables
# ============================================================

file_path = "student GPA data bank.xlsx"

# Select the dataset record to inspect (0-based index).
selected_row_idx = 210

# Value used to replace missing input features during prediction.
fill_value_for_missing_features = 8.0


# ============================================================
# Step 1: Load dataset
# ============================================================

df = pd.read_excel(file_path)

print("===== Dataset Information =====")
print(f"Number of records: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print("\nColumn names:")
print(list(df.columns))


# ============================================================
# Step 2: Auto-detect relevant columns
# ============================================================

# SSC and Intermediate columns
ssc_candidates = [c for c in df.columns if "ssc" in c.lower()]
inter_candidates = [c for c in df.columns if "inter" in c.lower()]

if not ssc_candidates:
    raise ValueError("SSC column could not be detected.")

if not inter_candidates:
    raise ValueError("Intermediate column could not be detected.")

ssc_col = ssc_candidates[0]
inter_col = inter_candidates[0]


# Detect semester columns automatically.
sem_cols_raw = [c for c in df.columns if "sem" in c.lower()]


def get_sem_number(name):
    """Return semester number from a column name."""
    name_low = name.lower()

    mapping = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
        "seventh": 7,
        "eighth": 8
    }

    for word, number in mapping.items():
        if word in name_low:
            return number

    digits = re.findall(r"\d+", name_low)

    if digits:
        return int(digits[0])

    return None


sem_cols_detected = []

for col in sem_cols_raw:
    sem_num = get_sem_number(col)

    if sem_num is not None:
        sem_cols_detected.append((sem_num, col))

sem_cols = [col for _, col in sorted(sem_cols_detected)]

if len(sem_cols) < 8:
    raise ValueError(
        f"Expected eight semester columns, but detected {len(sem_cols)}."
    )

print("\n===== Detected Columns =====")
print("SSC column       :", ssc_col)
print("Intermediate     :", inter_col)
print("Semester columns :", sem_cols)


# ============================================================
# Step 3: Clean duplicates and replace zero GPA values
# ============================================================

df = df.drop_duplicates().reset_index(drop=True)

print("\nRecords after dropping duplicates:", len(df))

# In this dataset, zero in a semester GPA is treated as a
# missing / failed-semester indicator rather than a valid GPA.
for col in sem_cols:
    df[col] = df[col].replace(0, np.nan)


# ============================================================
# Step 4: Identify Regular and Lateral students
# ============================================================

first_sem_col = sem_cols[0]
second_sem_col = sem_cols[1]

df["student_type"] = np.where(
    df[first_sem_col].isna() & df[second_sem_col].isna(),
    "Lateral",
    "Regular"
)

print("\n===== Student Type Counts =====")
print(df["student_type"].value_counts())


# ============================================================
# Step 5: Define input features and target semesters
# ============================================================

# Regular students:
# SSC + Intermediate + 1st to 4th semester GPA
regular_features = [
    ssc_col,
    inter_col,
    *sem_cols[0:4]
]

# Lateral students:
# SSC + Intermediate + 3rd and 4th semester GPA
lateral_features = [
    ssc_col,
    inter_col,
    *sem_cols[2:4]
]

# 5th to 8th semester GPAs are the prediction targets.
target_sems = sem_cols[4:]

print("\n===== Model Features =====")
print("Regular features:", regular_features)
print("Lateral features:", lateral_features)
print("Target semesters:", target_sems)


# ============================================================
# Step 6: Train separate models for each student type
#          and each target semester
# ============================================================

models = {}
model_scores = {}

print("\n===== Training Models =====")

for sem in target_sems:

    models[("Regular", sem)] = None
    models[("Lateral", sem)] = None

    # --------------------------------------------------------
    # Regular students
    # --------------------------------------------------------

    df_reg = df[df["student_type"] == "Regular"]

    if not df_reg.empty:

        X_regular = df_reg[regular_features]
        y_regular = df_reg[sem]

        mask_regular = (
            X_regular.notna().all(axis=1)
            & y_regular.notna()
        )

        if mask_regular.sum() > 0:

            X_train, X_test, y_train, y_test = train_test_split(
                X_regular[mask_regular],
                y_regular[mask_regular],
                test_size=0.2,
                random_state=42
            )

            model = LinearRegression()
            model.fit(X_train, y_train)

            models[("Regular", sem)] = model

            predictions = model.predict(X_test)
            score = r2_score(y_test, predictions)

            model_scores[("Regular", sem)] = score

            print(
                f"R² ({sem}, Regular): {score:.3f}"
            )

        else:
            print(f"R² ({sem}, Regular): Insufficient data")

    else:
        print(f"R² ({sem}, Regular): No Regular records")


    # --------------------------------------------------------
    # Lateral students
    # --------------------------------------------------------

    df_lat = df[df["student_type"] == "Lateral"]

    if not df_lat.empty:

        X_lateral = df_lat[lateral_features]
        y_lateral = df_lat[sem]

        mask_lateral = (
            X_lateral.notna().all(axis=1)
            & y_lateral.notna()
        )

        if mask_lateral.sum() > 0:

            X_train, X_test, y_train, y_test = train_test_split(
                X_lateral[mask_lateral],
                y_lateral[mask_lateral],
                test_size=0.2,
                random_state=42
            )

            model = LinearRegression()
            model.fit(X_train, y_train)

            models[("Lateral", sem)] = model

            predictions = model.predict(X_test)
            score = r2_score(y_test, predictions)

            model_scores[("Lateral", sem)] = score

            print(
                f"R² ({sem}, Lateral): {score:.3f}"
            )

        else:
            print(f"R² ({sem}, Lateral): Insufficient data")

    else:
        print(f"R² ({sem}, Lateral): No Lateral records")


# ============================================================
# Step 7: Predict for a specific dataset record
#          and compare Actual vs Predicted
# ============================================================

print("\n===== Specific Record Prediction =====")

if selected_row_idx < 0 or selected_row_idx >= len(df):
    raise IndexError(
        f"selected_row_idx {selected_row_idx} is out of range "
        f"(0 to {len(df) - 1})."
    )

record = df.iloc[selected_row_idx]

student_type = record["student_type"]

print(
    f"Selected record index: {selected_row_idx} "
    f"(Student Type: {student_type})"
)

if student_type == "Regular":
    features_for_model = regular_features
else:
    features_for_model = lateral_features


# Build input values and replace missing values.
input_values = {}

for feature in features_for_model:
    value = record.get(feature, np.nan)

    if pd.isna(value):
        value = fill_value_for_missing_features

    input_values[feature] = value


print(
    "\nInput features used for prediction "
    f"(missing values filled with {fill_value_for_missing_features}):"
)

for key, value in input_values.items():
    print(f"  {key}: {value}")


X_input = pd.DataFrame(
    [input_values],
    columns=features_for_model
)


for sem in target_sems:

    model = models.get((student_type, sem))

    if model is None:
        print(f"{sem}: No trained model - skipped")
        continue

    try:
        prediction = model.predict(X_input)[0]

    except Exception as error:
        print(f"{sem}: Prediction failed: {error}")
        continue

    actual = record.get(sem, np.nan)

    if pd.notna(actual):

        error = abs(prediction - actual)

        print(
            f"{sem}: "
            f"Actual = {actual:.3f}, "
            f"Predicted = {prediction:.3f}, "
            f"Absolute error = {error:.3f}"
        )

    else:

        print(
            f"{sem}: "
            f"Actual = NaN, "
            f"Predicted = {prediction:.3f} "
            f"(no actual value available)"
        )


# ============================================================
# Step 8: Static input prediction - Regular student
# ============================================================

print("\n===== Static Input Prediction: Regular Student =====")

# Edit these values for another Regular student.
ssc_value = 9.0
inter_value = 9.2
first_sem_value = 8.5
second_sem_value = 8.6
third_sem_value = 8.7
fourth_sem_value = 8.8

input_regular = pd.DataFrame([{
    ssc_col: ssc_value,
    inter_col: inter_value,
    sem_cols[0]: first_sem_value,
    sem_cols[1]: second_sem_value,
    sem_cols[2]: third_sem_value,
    sem_cols[3]: fourth_sem_value
}])


print("Assumed inputs (Regular):")

for key, value in input_regular.iloc[0].items():
    print(f"  {key}: {value}")


X_pred_regular = (
    input_regular[regular_features]
    .fillna(fill_value_for_missing_features)
)

print("\nPredicted targets (Regular):")

for sem in target_sems:

    model = models.get(("Regular", sem))

    if model is None:
        print(f"{sem}: No trained Regular model - skipped")
        continue

    try:
        prediction = model.predict(X_pred_regular)[0]

        print(
            f"{sem}: Predicted GPA = {prediction:.3f}"
        )

    except Exception as error:
        print(f"{sem}: Prediction failed: {error}")


# ============================================================
# Step 9: Static input prediction - Lateral student
# ============================================================

print("\n===== Static Input Prediction: Lateral Student =====")

# Edit these values for another Lateral student.
ssc_value_lat = 8.8
inter_value_lat = 9.0
third_sem_value_lat = 8.6
fourth_sem_value_lat = 8.7

input_lateral = pd.DataFrame([{
    ssc_col: ssc_value_lat,
    inter_col: inter_value_lat,
    sem_cols[2]: third_sem_value_lat,
    sem_cols[3]: fourth_sem_value_lat
}])


print("Assumed inputs (Lateral):")

for key, value in input_lateral.iloc[0].items():
    print(f"  {key}: {value}")


X_pred_lateral = (
    input_lateral[lateral_features]
    .fillna(fill_value_for_missing_features)
)

print("\nPredicted targets (Lateral):")

for sem in target_sems:

    model = models.get(("Lateral", sem))

    if model is None:
        print(f"{sem}: No trained Lateral model - skipped")
        continue

    try:
        prediction = model.predict(X_pred_lateral)[0]

        print(
            f"{sem}: Predicted GPA = {prediction:.3f}"
        )

    except Exception as error:
        print(f"{sem}: Prediction failed: {error}")


# ============================================================
# Step 10: Summary
# ============================================================

print("\n===== Project Completed =====")
print("Student types:", df["student_type"].value_counts().to_dict())
print("Target semesters:", target_sems)

print("\nR² Scores:")
for (student_type, semester), score in model_scores.items():
    print(f"  {semester} - {student_type}: {score:.3f}")
