import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load Data
file_path = "coffee_shop_sales.csv"  # Make sure this file exists
data = pd.read_csv(file_path)

# Preprocess Data
data['datetime'] = pd.to_datetime(data['datetime'])
data['hour'] = data['datetime'].dt.hour
data['coffee_name'] = data['coffee_name'].astype('category').cat.codes
data['cash_type'] = data['cash_type'].astype('category').cat.codes

# Define Features & Target
X = data[['coffee_name', 'cash_type', 'hour']]
y = data['money']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save Model
joblib.dump(model, "coffee_sales_model.pkl")
print("Model trained and saved successfully!")
