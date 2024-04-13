import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from pybaseball import statcast
import warnings
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Suppress FutureWarning from pybaseball
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the batter and pitcher stats from CSV files
batter_stats = pd.read_csv("batterstats.csv")
pitcher_stats = pd.read_csv("pitcherstats.csv")

# Retrieve the 2023 Statcast data
start_date = "2023-01-01"
end_date = "2023-12-31"
statcast_data = statcast(start_dt=start_date, end_dt=end_date)

# Filter the statcast data to only include occurrences where 'event' is not null
statcast_data = statcast_data[statcast_data['events'].notnull()]

# Merge the batter and pitcher stats with the statcast data using player_id
merged_data = pd.merge(statcast_data, batter_stats, left_on='batter', right_on='player_id')
merged_data = pd.merge(merged_data, pitcher_stats, left_on='pitcher', right_on='player_id')

print("merge_data columns:", merged_data.columns)

# Define features (batter and pitcher stats) and target variable (batter performance)
X = merged_data[['woba_x','woba_y','k_percent_x','k_percent_y','whiff_percent_x','whiff_percent_y','swing_percent_x','swing_percent_y','sweet_spot_percent_x','sweet_spot_percent_y','hard_hit_percent_x','hard_hit_percent_y','avg_best_speed_x','avg_best_speed_y']]  # Features
y = merged_data['events']  # Target variable

print(y)

# Assuming 'events' is a categorical variable with the names of baseball plays
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("y_encoded:", y_encoded)

print("Unique values in 'events' column:", merged_data['events'].unique())

print(merged_data['events'].value_counts())

print("Label encoding classes:", np.unique(y_encoded))



# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train an XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='merror')
xgb_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Save the model
xgb_model.save_model("batter_performance_xgb_model.xgb")

print("XGBoost model trained and saved successfully.")
