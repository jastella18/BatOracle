#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pybaseball import statcast, pitching_stats, batting_stats
import warnings
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle
from imblearn.over_sampling import SMOTE
import traceback
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Suppress FutureWarning from pybaseball
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the batter and pitcher stats from CSV files
#batter_stats = pd.read_csv("bstats.csv")
#pitcher_stats = pd.read_csv("pstats.csv")

# Retrieve the 2023 Statcast data
start_date = "2018-01-01"
end_date = "2024-04-20"
try:
    statcast_data = statcast(start_dt=start_date, end_dt=end_date)
    batter_stats = batting_stats(start_dt=start_date, end_dt=end_date)
    pitcher_stats = pitching_stats(start_dt=start_date, end_dt=end_date)
except Exception as e:
    print("An error occurred:", e)
    traceback.print_exc()

# Filter the statcast data to only include occurrences where 'event' is not null
statcast_data = statcast_data[statcast_data['events'].notnull()]
#%%
# Merge the batter and pitcher stats with the statcast data using player_id
merged_data = pd.merge(statcast_data, batter_stats, left_on=['batter','game_year'], right_on=['player_id','year'])
merged_data = pd.merge(merged_data, pitcher_stats, left_on=['pitcher', 'game_year'], right_on=['player_id','year'])
print("merge_data columns:", merged_data.columns)

# Define features (batter and pitcher stats) and target variable (home run or not)
X = merged_data[['woba_x', 'woba_y','slg_percent_x','slg_percent_y','batting_avg_x','batting_avg_y','oz_swing_percent_x','oz_swing_percent_y','home_run_x','home_run_y', 'k_percent_x', 'k_percent_y', 'whiff_percent_x', 'whiff_percent_y', 'swing_percent_x', 'swing_percent_y', 'sweet_spot_percent_x', 'sweet_spot_percent_y', 'hard_hit_percent_x', 'hard_hit_percent_y', 'avg_best_speed_x', 'avg_best_speed_y' , 'barrel_batted_rate_x', 'barrel_batted_rate_y', 'p_throws','stand']]  # Features
y = merged_data['events'].isin(['home_run', 'single', 'double', 'triple']).astype(int)  # Target variable (1 for hits, 0 for others)

merged_data = pd.get_dummies(merged_data, columns=['p_throws', 'stand'])
X = pd.get_dummies(X, columns=['p_throws', 'stand'])

with open ('merged_data.pkl', 'wb') as f:
    pickle.dump(merged_data, f)
#%%
import xgboost

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE oversampling
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

model = xgboost.XGBClassifier(n_estimators=3000, max_depth=15, learning_rate=0.05, random_state=42, n_jobs=-1)

model.fit(X_train_scaled, y_train_resampled)

###
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Perform k-fold cross-validation
k = 5  # Number of folds
cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train_scaled, y_train_resampled, cv=cv, scoring='f1')

# Print the cross-validation scores
print(f"Cross-Validation F1 Scores: {scores}")
print(f"Mean Cross-Validation F1 Score: {scores.mean():.2f}")
#%%
# Make predictions
y_pred = model.predict(X_test_scaled)
probabilities = model.predict_proba(X_test_scaled)
predictions_with_threshold = np.where(probabilities[:, 1] < 0.35, 0, 1)

# Evaluate the model
precision = precision_score(y_test, predictions_with_threshold)
recall = recall_score(y_test, predictions_with_threshold)
f1 = f1_score(y_test, predictions_with_threshold)
accuracy = accuracy_score(y_test, predictions_with_threshold)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
print(f"Accuracy: {accuracy:.2f}")

# Save the model
with open('batter_performance_xgb_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("model trained and saved successfully.")



# %%
import matplotlib.pyplot as plt
import numpy as np

# Get feature importances
importances = model.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [X_train.columns[i] for i in indices]

# Cumulative importances
cumulative_importances = np.cumsum(importances[indices])

# Make a line graph
plt.plot(range(len(importances)), cumulative_importances, 'g-')

# Draw line at 95% of importance retained
plt.hlines(y = 0.95, xmin=0, xmax=len(importances), color = 'r', linestyles = 'dashed')

# Format x ticks and labels
plt.xticks(range(len(importances)), names, rotation = 'vertical')

# Axis labels and title
plt.xlabel('Variable'); plt.ylabel('Cumulative Importance'); plt.title('Cumulative Importances');


#%%
import matplotlib.pyplot as plt
import numpy as np

# Get feature importances
importances = model.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [X_train.columns[i] for i in indices]

# Create plot
plt.figure(figsize=(10, 6))

# Create plot title
plt.title("Feature Importance")

# Add bars
plt.bar(range(X_train.shape[1]), importances[indices])

# Add feature names as x-axis labels
plt.xticks(range(X_train.shape[1]), names, rotation=90)

# Show plot
plt.show()


#%%
from sklearn.metrics import ConfusionMatrixDisplay 

# Assuming logistic_model is your model and X_test is defined
probabilities = model.predict_proba(X_test_scaled)
predictions_with_threshold = np.where(probabilities[:, 1] < .35, 0, 1)

ConfusionMatrixDisplay.from_predictions(y_test, predictions_with_threshold)
plt.title('Confusion Matrix with 35% Threshold')
plt.show()

#%%
from sklearn import tree

# Select one tree from the forest
chosen_tree = model.estimators_[5]

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(chosen_tree,
               feature_names = [X_train.columns[i] for i in indices], 
               class_names=['hit', 'no hit'],
               filled = True);

#%%
from sklearn.metrics import plot_precision_recall_curve

plot_precision_recall_curve(model, X_test_scaled, y_test)
plt.show()

# %%
