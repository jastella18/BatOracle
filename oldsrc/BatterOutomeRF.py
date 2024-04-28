from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def load_model_and_predict(batter_id, pitcher_id, batter_stats, pitcher_stats):
    # Load the saved Random Forest model
    with open('src/batter_performance_rf_model.pkl', 'rb') as file:
        batter_performance_model = pickle.load(file)

    # Get the batter and pitcher statistics
    batter_data = batter_stats[batter_stats['last_name, first_name'] == batter_id]
    pitcher_data = pitcher_stats[pitcher_stats['last_name, first_name'] == pitcher_id]

    if batter_data.empty or pitcher_data.empty:
        print("Invalid batter or pitcher ID provided. Please try again.")
        return

    # Prepare the feature data
    X_test = pd.DataFrame({
        'woba_x': [batter_data['woba'].values[0]],
        'woba_y': [pitcher_data['woba'].values[0]],
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
        'avg_best_speed_y': [pitcher_data['avg_best_speed'].values[0]]
    }, index=[0])

    # Make predictions
    y_pred = batter_performance_model.predict(X_test)

    
    print("Probaility:", y_pred_proba)

    return y_pred

if __name__ == "__main__":
    # Load the batter and pitcher statistics
    batter_stats = pd.read_csv("src/batterstats.csv")
    pitcher_stats = pd.read_csv("src/pitcherstats.csv")

    # Prompt the user for batter and pitcher IDs
    batter_id = input("Enter the batter ID: ")
    pitcher_id = input("Enter the pitcher ID: ")

    # Load the model and make predictions
    prediction = load_model_and_predict(batter_id, pitcher_id, batter_stats, pitcher_stats)
    

    if prediction:
        print(f"Predicted batter performance: {prediction}")