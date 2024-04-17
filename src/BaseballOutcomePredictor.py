from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.isotonic import IsotonicRegression



def load_model_and_predict(batter_id, pitcher_id, batter_stats, pitcher_stats):
    # Load the saved Logistic Regression model
    with open('src/batter_performance_logreg_model.pkl', 'rb') as file:
        batter_performance_model = pickle.load(file)
    
    with open('src/merged_data.pkl', 'rb') as file:
        merged_data = pickle.load(file)

    with open('src/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)


    batter_stats_2024 = batter_stats[batter_stats['year'] == 2024]
    pitcher_stats_2024 = pitcher_stats[pitcher_stats['year'] == 2024]

    # Get the batter and pitcher statistics
    batter_data = batter_stats_2024[batter_stats_2024['last_name, first_name'] == batter_id]
    pitcher_data = pitcher_stats_2024[pitcher_stats_2024['last_name, first_name'] == pitcher_id]

    if batter_data.empty or pitcher_data.empty:
        print("Invalid or Inactive batter or pitcher ID provided. Please try again.")
        return

    # Prepare the feature data
    X_test = pd.DataFrame({
        'woba_x': [batter_data['woba'].values[0]],
        'woba_y': [pitcher_data['woba'].values[0]],
        'home_run_x': [batter_data['home_run'].values[0]],
        'home_run_y': [pitcher_data['home_run'].values[0]],
        'k_percent_x': [batter_data['k_percent'].values[0]],
        'k_percent_y': [pitcher_data['k_percent'].values[0]],
        'whiff_percent_x': [batter_data['whiff_percent'].values[0]],
        'whiff_percent_y': [pitcher_data['whiff_percent'].values[0]],
        'swing_percent_x': [batter_data['swing_percent'].values[0]],
        'swing_percent_y': [pitcher_data['swing_percent'].values[0]],
        'sweet_spot_percent_x': [batter_data['sweet_spot_percent'].values[0]],
        'sweet_spot_percent_y': [pitcher_data['sweet_spot_percent'].values[0]],
        'hard_hit_percent_x': [batter_data['hard_hit_percent'].values[0]],
        'hard_hit_percent_y': [pitcher_data['hard_hit_percent'].values[0]],
        'avg_best_speed_x': [batter_data['avg_best_speed'].values[0]],
        'avg_best_speed_y': [pitcher_data['avg_best_speed'].values[0]],
        'barrel_batted_rate_x': [batter_data['barrel_batted_rate'].values[0]],
        'barrel_batted_rate_y': [pitcher_data['barrel_batted_rate'].values[0]],
        'p_throws_L': [merged_data['p_throws_L'].values[0]],
        'p_throws_R': [merged_data['p_throws_R'].values[0]],
        'stand_L': [merged_data['stand_L'].values[0]],
        'stand_R': [merged_data['stand_R'].values[0]]

    }, index=[0])

    X_test_scaled = scaler.transform(X_test)


    # Make predictions
    y_pred = batter_performance_model.predict(X_test_scaled)

    y_pred_proba_uncalibrated = batter_performance_model.predict_proba(X_test_scaled)[:, 1]

    """   
    with open('src/calibrator.pkl', 'rb') as file:
        loaded_calibrator = pickle.load(file)

    # Now you can use `loaded_calibrator` to calibrate your predictions
    y_pred_proba_calibrated = loaded_calibrator.transform(y_pred_proba_uncalibrated)"""

    print("Probability of batter hitting a homerun: {:.2f}%".format(y_pred_proba_uncalibrated[0] * 100))

    return y_pred

if __name__ == "__main__":
    # Load the batter and pitcher statistics
    batter_stats = pd.read_csv("src/bstats.csv")
    pitcher_stats = pd.read_csv("src/pstats.csv")

    # Prompt the user for batter and pitcher IDs
    batter_id = input("Enter the batter ID: ")
    pitcher_id = input("Enter the pitcher ID: ")
    

    # Load the model and make predictions
    prediction = load_model_and_predict(batter_id, pitcher_id, batter_stats, pitcher_stats)

    if prediction:
        print(f"Predicted batter performance: {prediction}")
