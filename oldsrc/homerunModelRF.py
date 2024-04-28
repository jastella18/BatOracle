#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pybaseball import statcast
import warnings
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
from imblearn.over_sampling import SMOTE

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

# Define features (batter and pitcher stats) and target variable (home run or not)
X = merged_data[['woba_x', 'woba_y', 'k_percent_x', 'k_percent_y', 'whiff_percent_x', 'whiff_percent_y', 'swing_percent_x', 'swing_percent_y', 'sweet_spot_percent_x', 'sweet_spot_percent_y', 'hard_hit_percent_x', 'hard_hit_percent_y', 'avg_best_speed_x', 'avg_best_speed_y']]  # Features
y = (merged_data['events'] == 'home_run').astype(int)  # Target variable (1 for home run, 0 for others)
#%%
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE oversampling
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train_resampled)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Evaluate the model
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# Save the model
with open('batter_performance_rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

print("Random Forest model trained and saved successfully.")
#%%
import matplotlib.pyplot as plt

# Get feature importances
importances = rf_model.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Print the feature rankings
print("Feature importances:")
for f in range(X.shape[1]):
    print("%d. %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))

# Plot the feature importances
plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], color="r", align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()
# %%
import matplotlib.pyplot as plt


y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_proba, alpha=0.5)
plt.xlabel('True Labels')
plt.ylabel('Predicted Probabilities')
plt.title('Scatter Plot of Predictions')

# Add a diagonal line for reference
plt.plot([0, 1], [0, 1], 'r--', label='Perfect Predictions')
plt.legend()
plt.show()
# %%
