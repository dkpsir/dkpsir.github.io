# Project 3: Student Pass / Fail Prediction using Logistic Regression

# Step 1: Import libraries
import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)


# Step 2: Load the CSV dataset
# Keep "Student_Pass_Fail_Data Set.csv" in the same folder as this Python file.
data_path = "Student_Pass_Fail_Data Set.csv"
df = pd.read_csv(data_path)


# Step 3: Basic information
print("\n===== First Few Rows =====")
print(df.head())

print("\n===== Dataset Information =====")
df.info()

print("\n===== Missing Values =====")
print(df.isnull().sum())


# Step 4: Remove Student_ID
# Student_ID is unique for each student and is not useful for prediction.
if "Student_ID" in df.columns:
    df.drop("Student_ID", axis=1, inplace=True)


# Step 5: Convert categorical columns to category type
categorical_cols = [
    "Gender",
    "Parental_Education_Level",
    "Internet_Access_at_Home",
    "Extracurricular_Activities",
    "Pass_Fail"
]

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype("category")


print("\n===== Data Types After Preprocessing =====")
print(df.dtypes)


# Step 6: Exploratory Data Analysis

numeric_cols = [
    "Study_Hours_per_Week",
    "Attendance_Rate",
    "Past_Exam_Scores",
    "Final_Exam_Score"
]

# Distribution of numerical variables
for col in numeric_cols:
    plt.figure(figsize=(7, 4))
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# Correlation heatmap
numeric_df = df.select_dtypes(include=[np.number])

if numeric_df.shape[1] >= 2:
    plt.figure(figsize=(8, 6))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


# Pair plot for numerical variables
sns.pairplot(df[numeric_cols])
plt.show()


# Categorical distributions
eda_categorical_cols = [
    "Gender",
    "Parental_Education_Level",
    "Internet_Access_at_Home",
    "Extracurricular_Activities",
    "Pass_Fail"
]

for col in eda_categorical_cols:
    plt.figure(figsize=(7, 4))
    sns.countplot(x=col, data=df)
    plt.title(f"Count Plot of {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Step 7: One-hot encode categorical features
df_encoded = pd.get_dummies(df, drop_first=True)

print("\n===== Encoded Features =====")
print(df_encoded.head())

print("\nDataFrame shape after encoding:", df_encoded.shape)


# Step 8: Define input features and target

# The target becomes numeric after one-hot encoding.
# For the source dataset, Pass_Fail is represented as Pass/Fail.
pass_fail_cols = [col for col in df_encoded.columns if col.startswith("Pass_Fail_")]

if len(pass_fail_cols) == 1:
    target_col = pass_fail_cols[0]

    # One encoded column is sufficient because drop_first=True.
    # Map the encoded target to 0/1.
    y = df_encoded[target_col].astype(int)
    X = df_encoded.drop(columns=[target_col])

else:
    # Fallback for a dataset where Pass_Fail is already numeric.
    X = df_encoded.drop(columns=["Pass_Fail"])
    y = df_encoded["Pass_Fail"].astype(int)


# Step 9: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)


# Step 10: Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# Step 11: Make predictions
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]


# Step 12: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\n===== Model Evaluation =====")
print(f"Prediction Accuracy: {accuracy:.4f}")

print("\n===== Classification Report =====")
print(classification_report(y_test, y_pred))


# Step 13: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\n===== Confusion Matrix =====")
print(cm)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# Step 14: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

print(f"\nROC-AUC: {roc_auc:.4f}")

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# Step 15: Prediction for a randomly selected test student
random_index = random.randint(0, len(X_test) - 1)

random_record = X_test.iloc[[random_index]]
actual_result = y_test.iloc[random_index]
predicted_result = model.predict(random_record)[0]

print("\n===== Random Student Prediction =====")
print("Student Features:")
print(random_record)

print(
    "\nActual Result:",
    "Pass" if actual_result == 1 else "Fail"
)

print(
    "Predicted Result:",
    "Pass" if predicted_result == 1 else "Fail"
)


# Step 16: Prediction for a manually assumed student

assumed_data = {
    "Study_Hours_per_Week": 28,
    "Attendance_Rate": 89.5,
    "Past_Exam_Scores": 75,
    "Final_Exam_Score": 65,
    "Gender_Male": 1,
    "Parental_Education_Level_High School": 0,
    "Parental_Education_Level_Masters": 1,
    "Parental_Education_Level_PhD": 0,
    "Internet_Access_at_Home_Yes": 1,
    "Extracurricular_Activities_Yes": 0
}

input_df = pd.DataFrame([assumed_data])

# Make sure the prediction data has exactly the same
# columns and order used during model training.
input_df = input_df.reindex(columns=X.columns, fill_value=0)

predicted_value = model.predict(input_df)[0]

print("\n===== Manually Assumed Student =====")
for key, value in assumed_data.items():
    print(f"{key} : {value}")

print(
    "\nPrediction Result:",
    "Pass" if predicted_value == 1 else "Fail"
)
