import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from pybaseball import statcast
import warnings
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

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
#%%
# Merge the batter and pitcher stats with the statcast data using player_id
merged_data = pd.merge(statcast_data, batter_stats, left_on='batter', right_on='player_id')
merged_data = pd.merge(merged_data, pitcher_stats, left_on='pitcher', right_on='player_id')

print("merge_data columns:", merged_data.columns)

# Define features (batter and pitcher stats) and target variable (batter performance)
X = merged_data[['woba_x','woba_y','k_percent_x','k_percent_y','whiff_percent_x','whiff_percent_y','swing_percent_x','swing_percent_y','sweet_spot_percent_x','sweet_spot_percent_y','hard_hit_percent_x','hard_hit_percent_y','avg_best_speed_x','avg_best_speed_y']]  # Features
y = merged_data['events']  # Target variable

print("y:", y)

# Assuming 'events' is a categorical variable with the names of baseball plays
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save the fitted LabelEncoder
with open('label_encoder_names.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)


print("y_encoded:", y_encoded)

# Decode each value in y_encoded
y_decoded = label_encoder.inverse_transform(y_encoded)

# Print the decoded values
print("Decoded values:", y_decoded)

print("Unique values in 'events' column:", merged_data['events'].unique())

print(merged_data['events'].value_counts())

print("Label encoding classes:", np.unique(y_encoded))

unique, counts = np.unique(y_encoded, return_counts=True)
print("Counts of each class in y_encoded:", dict(zip(unique, counts)))

# Remove values from y_encoded with 5 or fewer occurrences
values_to_remove = [2, 3, 14, 15, 16, 19, 24]
mask = ~np.isin(y_encoded, values_to_remove)
y_encoded_filtered = y_encoded[mask]

# Filter X
X_filtered = X[mask]

# Remap the class labels in y_encoded_filtered to a contiguous range
label_encoder = LabelEncoder()
y_encoded_filtered = label_encoder.fit_transform(y_encoded_filtered)

# Save the fitted LabelEncoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Perform the train-test split on the filtered data
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_encoded_filtered, test_size=0.1, random_state=42)


print("Unique values in y_train:", np.unique(y_train))
print("Unique values in y_encoded:", np.unique(y_encoded_filtered))

print("Counts of each class in y_encoded:", dict(zip(unique, counts)))

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train an XGBoost model
unique_classes = np.unique(y_encoded)
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

# %%
