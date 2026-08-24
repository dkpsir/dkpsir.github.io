# Project 2: Weather Prediction using Linear Regression

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error


# Step 2: Load Excel dataset
# Keep "weather data set.xlsx" in the same folder as this Python file.
file_path = "weather data set.xlsx"
data = pd.read_excel(file_path)


# Step 3: Basic information
print("\n===== Dataset Info =====")
print(data.info())

print("\n===== Missing Values (before cleaning) =====")
print(data.isnull().sum())


# Step 4: Preprocessing
data["date_of_record"] = pd.to_datetime(data["date_of_record"])
data = data.sort_values("date_of_record")

# Extract date components
data["year"] = data["date_of_record"].dt.year
data["month_num"] = data["date_of_record"].dt.month
data["day"] = data["date_of_record"].dt.day

# Encode station name
le_station = LabelEncoder()
data["station_encoded"] = le_station.fit_transform(data["station_name"])


# Step 5: Handle missing values
temp_cols = ["avg_temp", "min_temp", "max_temp"]
rain_col = "rainfall"

for col in temp_cols:
    data[col] = data[col].interpolate(
        method="linear",
        limit_direction="both",
        limit=2
    )
    data[col] = data[col].fillna(data[col].mean())

# Fill missing rainfall with 0
data[rain_col] = data[rain_col].fillna(0)

print("\n===== Missing Values (after cleaning) =====")
print(data.isnull().sum())


# Step 6: Define input and output features
X = data[["year", "month_num", "day", "station_encoded"]]
y = data[["avg_temp", "min_temp", "max_temp", "rainfall"]]


# Step 7: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Step 8: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)


# Step 9: Evaluate the model
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n===== Model Evaluation =====")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")


# Step 10: Define prediction function
def predict_weather(future_date, station_name):
    """
    Predict average temperature, minimum temperature,
    maximum temperature and rainfall for a given
    date and station.

    If the date and station exist in the dataset,
    percentage errors between actual and predicted
    values are also calculated.
    """

    future_date = pd.to_datetime(future_date)

    year = future_date.year
    month_num = future_date.month
    day = future_date.day

    try:
        station_encoded = le_station.transform([station_name])[0]
    except ValueError:
        print(f"Station '{station_name}' not found in training data.")
        return None

    # Prepare feature vector
    X_future = pd.DataFrame(
        [[year, month_num, day, station_encoded]],
        columns=["year", "month_num", "day", "station_encoded"]
    )

    # Generate prediction
    prediction = model.predict(X_future)

    output_vars = [
        "avg_temp",
        "min_temp",
        "max_temp",
        "rainfall"
    ]

    pred_dict = dict(zip(output_vars, prediction[0]))

    # Check whether actual data exists
    existing = data[
        (data["station_name"] == station_name) &
        (data["date_of_record"] == future_date)
    ]

    if not existing.empty:

        actual = existing[output_vars].iloc[0]
        errors = {}

        for col in output_vars:
            actual_val = actual[col]
            pred_val = pred_dict[col]

            if actual_val == 0:
                error_pct = 0
            else:
                error_pct = (
                    (pred_val - actual_val) / actual_val
                ) * 100

            errors[col + "_error_%"] = round(error_pct, 2)

        result = {**pred_dict, **errors}

        print(
            f"\nActual data found for "
            f"{station_name} on {future_date.date()}"
        )

        return pd.DataFrame([result])

    else:
        print(
            f"\nNo actual data available for "
            f"{station_name} on {future_date.date()}"
        )

        return pd.DataFrame([pred_dict])


# Step 11: Example prediction
print("\n===== Example Prediction =====")

future_date = "2024-12-15"
station = "Vijayawada"

result = predict_weather(future_date, station)
print(result)


# Step 12: Display model score
print(f"\nModel Score (R²): {model.score(X_test, y_test):.4f}")


# Optional: Simple actual vs predicted plot for average temperature
plt.figure(figsize=(8, 5))
plt.scatter(
    y_test["avg_temp"],
    y_pred[:, 0],
    alpha=0.35
)
plt.xlabel("Actual Average Temperature (°C)")
plt.ylabel("Predicted Average Temperature (°C)")
plt.title("Actual vs Predicted Average Temperature")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
