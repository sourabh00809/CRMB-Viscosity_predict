import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

data = pd.read_csv("revised_crmb1.csv")

features = ["cr_dosage", "mixing_temp", "mixing_time", "cr_size", "base_viscosity", "RPM"]
target = "crmb_viscosity"

X = data[features]
y = data[target]

#training 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#model summery
y_pred = model.predict(X_test)
print("Model summery:")
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.3f}")


joblib.dump(model, "crmb_viscosity_rf_model.pkl")

#prediction for any new sample input   0.39333  0.23536
new_sample = pd.DataFrame({
    "cr_dosage": [20],
    "mixing_temp": [180],
    "mixing_time": [1],
    "cr_size": [30],
    "base_viscosity": [0.39333],
    "RPM": [1000]
})

predicted_viscosity = model.predict(new_sample)[0]
print(f"CRMB Viscosity for given sample data(ranfor): {predicted_viscosity:.2f}")
