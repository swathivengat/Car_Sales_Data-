import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("D:\◤𝔖𝔴𝔞𝔱𝔥𝔦◢\car_sales_data.csv")
print("First 5 rows:\n", df.head())
print("\nDataset Info:\n")
print(df.info())
print("\nDataset Summary:\n")
print(df.describe())

X = df.drop("Price", axis=1)
y = df["Price"]

cat_cols = ["Manufacturer", "Model", "Fuel type"]
num_cols = ["Engine size", "Year of manufacture", "Mileage"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nEvaluation Results:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

sample = pd.DataFrame([{
    "Manufacturer": "Toyota",
    "Model": "Yaris",
    "Engine size": 1.2,
    "Fuel type": "Petrol",
    "Year of manufacture": 2015,
    "Mileage": 50000
}])

predicted_price = model.predict(sample)[0]
print("\nPredicted Price for sample car:", predicted_price)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors="k")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         "r--", lw=2, label="Perfect Prediction")

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.legend()
plt.grid(True)
plt.show()
