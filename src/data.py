import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from pybaseball import statcast
import warnings

# Suppress FutureWarning from pybaseball
warnings.simplefilter(action='ignore', category=FutureWarning)


batter_stats = pd.read_csv("batterstats.csv")
pitcher_stats = pd.read_csv("pitcherstats.csv")

# Retrieve the 2023 Statcast data
start_date = "2023-01-01"
end_date = "2023-12-31"
statcast_data = statcast(start_dt=start_date, end_dt=end_date)

# Define features (batter and pitcher stats) and target variable (batter performance)
X = batter_stats["woba"],pitcher_stats["woba"]  # Features
y = statcast_data["events"]  # Target variable

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train an XGBoost model
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Save the model
xgb_model.save_model("batter_performance_xgb_model.xgb")

print("XGBoost model trained and saved successfully.")
