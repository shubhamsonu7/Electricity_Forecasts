#           ELECTRICITY BILL FORECASTER

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. CREATE MONTHLY HISTORICAL ELECTRICITY DATA

np.random.seed(42)

months = pd.date_range(
    start="2022-01-01",
    periods=48,
    freq="MS"
)

data = pd.DataFrame({
    "Month": months
})

# Month number
data["Month_Number"] = data["Month"].dt.month

# Average temperature according to month
temperature = {
    1: 22, 2: 25, 3: 28, 4: 31,
    5: 31, 6: 30, 7: 28, 8: 28,
    9: 28, 10: 27, 11: 24, 12: 22
}

data["Temperature"] = data["Month_Number"].map(temperature)

# Number of occupants
data["Occupants"] = 4

# Number of appliances
data["Appliances"] = 10

# Average daily usage hours
data["Daily_Usage_Hours"] = np.random.uniform(
    5, 9, len(data)
)


# 2. GENERATE HISTORICAL MONTHLY CONSUMPTION

consumption = []

previous = 180

for i in range(len(data)):

    temp = data.loc[i, "Temperature"]
    usage = data.loc[i, "Daily_Usage_Hours"]

    current = (
        50 
        + (temp * 3)
        + (usage * 15)
        + np.random.normal(0, 15)
        + previous * 0.35
    )

    current = max(80, current)

    consumption.append(current)

    previous = current


data["Consumption_kWh"] = consumption


# 3. CREATE PREVIOUS MONTH CONSUMPTION

data["Previous_Consumption"] = (
    data["Consumption_kWh"].shift(1)
)

# Remove first row because it has no previous month
data = data.dropna().reset_index(drop=True)


# 4. ELECTRICITY BILL CALCULATION

# These are SAMPLE tariff rates for demonstration.
# Replace with the applicable official tariff if required.

def calculate_bill(units):

    if units <= 100:
        return units * 3

    elif units <= 200:
        return (
            100 * 3
            + (units - 100) * 4
        )

    elif units <= 300:
        return (
            100 * 3
            + 100 * 4
            + (units - 200) * 5
        )

    else:
        return (
            100 * 3
            + 100 * 4
            + 100 * 5
            + (units - 300) * 6
        )


data["Bill_INR"] = data["Consumption_kWh"].apply(
    calculate_bill
)


# 5. DISPLAY HISTORICAL DATA

print("\n===================================================")
print("       HISTORICAL MONTHLY ELECTRICITY DATA")
print("===================================================")

print(
    data[
        [
            "Month",
            "Temperature",
            "Daily_Usage_Hours",
            "Previous_Consumption",
            "Consumption_kWh",
            "Bill_INR"
        ]
    ].tail(12)
)


# 6. PLOT MONTHLY CONSUMPTION

plt.figure(figsize=(12, 5))

plt.plot(
    data["Month"],
    data["Consumption_kWh"],
    marker="o"
)

plt.title("Monthly Electricity Consumption")
plt.xlabel("Month")
plt.ylabel("Consumption (kWh)")

plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()


# 7. PREPARE DATA FOR MACHINE LEARNING

features = [
    "Month_Number",
    "Temperature",
    "Occupants",
    "Appliances",
    "Daily_Usage_Hours",
    "Previous_Consumption"
]

X = data[features]

y = data["Consumption_kWh"]


# 8. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# 9. TRAIN MULTIPLE LINEAR REGRESSION MODEL

model = LinearRegression()

model.fit(
    X_train,
    y_train
)


# 10. TEST MODEL

y_pred = model.predict(X_test)


# 11. MODEL PERFORMANCE

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)

print("\n===================================================")
print("             MODEL PERFORMANCE")
print("===================================================")

print(f"MAE  : {mae:.2f} kWh")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f} kWh")
print(f"R2   : {r2:.4f}")


# 12. PREDICT NEXT MONTH

last_row = data.iloc[-1]

last_month = last_row["Month"]

# Next month
next_month = last_month + pd.DateOffset(months=1)

next_month_number = next_month.month

next_temperature = temperature[next_month_number]

next_usage_hours = (
    data["Daily_Usage_Hours"].tail(3).mean()
)

previous_consumption = (
    last_row["Consumption_kWh"]
)


next_month_data = pd.DataFrame({

    "Month_Number": [next_month_number],

    "Temperature": [next_temperature],

    "Occupants": [4],

    "Appliances": [10],

    "Daily_Usage_Hours": [
        next_usage_hours
    ],

    "Previous_Consumption": [
        previous_consumption
    ]
})


# 13. FORECAST NEXT MONTH'S CONSUMPTION

predicted_consumption = model.predict(
    next_month_data
)[0]

predicted_consumption = max(
    0,
    predicted_consumption
)


# 14. FORECAST NEXT MONTH'S ELECTRICITY BILL

predicted_bill = calculate_bill(
    predicted_consumption
)


# 15. FINAL RESULT

print("\n===================================================")
print("          NEXT MONTH ELECTRICITY FORECAST")
print("===================================================")

print(
    "Forecast Month        :",
    next_month.strftime("%B %Y")
)

print(
    f"Predicted Consumption : "
    f"{predicted_consumption:.2f} kWh"
)

print(
    f"Estimated Electricity Bill : "
    f"₹{predicted_bill:.2f}"
)

print("===================================================")

# 16. ACTUAL VS PREDICTED GRAPH

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Consumption (kWh)")
plt.ylabel("Predicted Consumption (kWh)")

plt.title(
    "Actual vs Predicted Electricity Consumption"
)

plt.grid(True)

plt.tight_layout()
plt.show()