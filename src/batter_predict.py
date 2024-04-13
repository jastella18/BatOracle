import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_model_and_predict(batter_id, pitcher_id, batter_stats, pitcher_stats):
    # Load the saved XGBoost model
    batter_performance_model = xgb.XGBClassifier()
    batter_performance_model.load_model("batter_performance_xgb_model.xgb")

    # Get the batter and pitcher statistics
    batter_data = batter_stats[batter_stats['last_name, first_name'] == batter_id]
    pitcher_data = pitcher_stats[pitcher_stats['last_name, first_name'] == pitcher_id]

    if batter_data.empty or pitcher_data.empty:
        print("Invalid batter or pitcher ID provided. Please try again.")
        return

    # Prepare the feature data
    X_test = pd.DataFrame({
        'woba_x': batter_data['woba'],
        'woba_y': pitcher_data['woba'],
        'k_percent_x': batter_data['k_percent'],
        'k_percent_y': pitcher_data['k_percent'],
        'whiff_percent_x': batter_data['whiff_percent'],
        'whiff_percent_y': pitcher_data['whiff_percent'],
        'swing_percent_x': batter_data['swing_percent'],
        'swing_percent_y': pitcher_data['swing_percent'],
        'sweet_spot_percent_x': batter_data['sweet_spot_percent'],
        'sweet_spot_percent_y': pitcher_data['sweet_spot_percent'],
        'hard_hit_percent_x': batter_data['hard_hit_percent'],
        'hard_hit_percent_y': pitcher_data['hard_hit_percent'],
        'avg_best_speed_x': batter_data['avg_best_speed'],
        'avg_best_speed_y': pitcher_data['avg_best_speed']
    }, index=[0])

    # Make predictions
    y_pred = batter_performance_model.predict(X_test)

    # Decode the predicted labels back to their original string values
    label_encoder = LabelEncoder()
    label_encoder.classes_ = ['single', 'double', 'triple', 'home_run', 'strike_out', 'walk', 'field_out','force_out','ground_out','fly_out','line_out','pop_out','sac_bunt','sac_fly', 'hit_by_pitch', 'intentional_walk', 'caught_stealing','pickoff','wild_pitch','passed_ball','balk','error','fielders_choice','catcher_interference','batter_interference','runner_interference','fan_interference']  # Provide the list of label names
    y_pred_decoded = label_encoder.inverse_transform(y_pred)

    return y_pred_decoded[0]

if __name__ == "__main__":
    # Load the batter and pitcher statistics
    batter_stats = pd.read_csv("batterstats.csv")
    pitcher_stats = pd.read_csv("pitcherstats.csv")
    
    # Prompt the user for batter and pitcher IDs
    batter_id = input("Enter the batter ID: ")
    pitcher_id = input("Enter the pitcher ID: ")
    
        
    print("Batter player_id values:")
    print(batter_stats['player_id'].unique())
    print("\nPitcher player_id values:")
    print(pitcher_stats['player_id'].unique())
    
    print(pitcher_stats['player_id'] == pitcher_id)

    # Load the model and make predictions
    prediction = load_model_and_predict(batter_id, pitcher_id, batter_stats, pitcher_stats)

    if prediction:
        print(f"Predicted batter performance: {prediction}")
