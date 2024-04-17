#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pybaseball import statcast
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
batter_stats = pd.read_csv("bstats.csv")
pitcher_stats = pd.read_csv("pstats.csv")

# Retrieve the 2023 Statcast data
start_date = "2021-01-01"
end_date = "2024-04-15"
try:
    statcast_data = statcast(start_dt=start_date, end_dt=end_date)
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
X = merged_data[['woba_x', 'woba_y','home_run_x','home_run_y', 'k_percent_x', 'k_percent_y', 'whiff_percent_x', 'whiff_percent_y', 'swing_percent_x', 'swing_percent_y', 'sweet_spot_percent_x', 'sweet_spot_percent_y', 'hard_hit_percent_x', 'hard_hit_percent_y', 'avg_best_speed_x', 'avg_best_speed_y' , 'barrel_batted_rate_x', 'barrel_batted_rate_y']]  # Features
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

# Train a logistic regression model
logistic_model = LogisticRegression(class_weight='balanced')

from sklearn.model_selection import StratifiedKFold, cross_val_score
# Perform k-fold cross-validation
k = 5  # Number of folds
cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
scores = cross_val_score(logistic_model, X_train_scaled, y_train_resampled, cv=cv, scoring='f1')

# Print the cross-validation scores
print(f"Cross-Validation F1 Scores: {scores}")
print(f"Mean Cross-Validation F1 Score: {scores.mean():.2f}")


logistic_model.fit(X_train_scaled, y_train_resampled)


# Make predictions
y_pred = logistic_model.predict(X_test_scaled)

# Evaluate the model
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")



# Save the model
with open('batter_performance_logreg_model.pkl', 'wb') as f:
    pickle.dump(logistic_model, f)

print("Logistic Regression model trained and saved successfully.")



# %%
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

y_pred_proba = logistic_model.predict_proba(X_test_scaled)

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
# %%
