import pandas as pd
from pybaseball import statcast, playerid_reverse_lookup
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import re

# Save preprocessed data to a CSV file
def save_preprocessed_data(data, filename):
    data.to_csv(filename, index=False)

# Load preprocessed data from a CSV file
def load_preprocessed_data(filename):
    return pd.read_csv(filename)

def get_player_data(start_year, end_year):
    # Load preprocessed data if available
    try:
        player_data = load_preprocessed_data('preprocessed_data.csv')
    except FileNotFoundError:
        player_data = pd.DataFrame()
        for year in range(start_year, end_year + 1):
            data = statcast(start_dt=f'{year}-01-01', end_dt=f'{year}-12-31')
            selected_data = data[['batter', 'pitcher', 'events', 'pitch_type', 'release_speed', 'estimated_woba_using_speedangle', 'estimated_ba_using_speedangle']]
            player_data = pd.concat([player_data, selected_data], ignore_index=True)
        # Save preprocessed data
        save_preprocessed_data(player_data, 'preprocessed_data.csv')
    return player_data

def create_batter_dataset(player_data):
    # Convert pitch types to their actual names
    pitch_type_map = {
        0: 'Unknown',
        1: 'FF', 2: 'FC', 3: 'FS', 4: 'FT',
        5: 'CH', 6: 'CU', 7: 'KC', 8: 'EP',
        9: 'SI', 10: 'SL', 11: 'FO', 12: 'PO',
        13: 'IN', 14: 'SC', 15: 'KN', 16: 'UN',
        17: 'RB', 18: 'PT', 19: 'CS', 20: 'RS'
    }

    label_encoder = LabelEncoder()
    player_data['pitch_type'] = label_encoder.fit_transform(player_data['pitch_type'])
    player_data['pitch_type'] = player_data['pitch_type'].map(pitch_type_map)

    # Keep only hit events
    hit_events = ['single', 'double', 'triple', 'home_run']
    player_data = player_data[player_data['events'].isin(hit_events)]

    # Convert batter and pitcher IDs to names
    batter_names = playerid_reverse_lookup(player_data['batter'].unique(), key_type='mlbam')
    player_data = player_data.merge(batter_names[['key_mlbam', 'key_bbref']], left_on='batter', right_on='key_mlbam', how='left')
    player_data.drop('batter', axis=1, inplace=True)
    player_data.rename(columns={'key_bbref': 'batter'}, inplace=True)

    pitcher_names = playerid_reverse_lookup(player_data['pitcher'].unique(), key_type='retro')
    player_data = player_data.merge(pitcher_names[['key_retro', 'key_bbref']], left_on='pitcher', right_on='key_retro', how='left')
    player_data.drop('pitcher', axis=1, inplace=True)
    player_data.rename(columns={'key_bbref': 'pitcher'}, inplace=True)

    # Calculate ERA against certain ranges of ERAs
    era_bins = [0, .2, .3, .4, .6, float('inf')]
    era_labels = ['Excellent', 'Good', 'Average', 'Poor', 'Very Poor']
    player_data['ERA_range'] = pd.cut(player_data['estimated_ba_using_speedangle'], bins=era_bins, labels=era_labels, include_lowest=True)
    era_freq = player_data.groupby(['batter', 'ERA_range'])['events'].count().unstack('ERA_range', fill_value=0)
    era_freq.columns = [f'ERA_{col}' for col in era_freq.columns]

    # Calculate frequency hitting each pitch type
    pitch_freq = player_data.groupby(['batter', 'pitch_type'])['events'].count().unstack('pitch_type', fill_value=0)
    pitch_freq.columns = [f'pitch_type_{col}' for col in pitch_freq.columns]

    # Calculate frequency hitting certain velocity ranges
    velocity_bins = [80, 90, 95, 100, float('inf')]
    velocity_labels = ['Slow', 'Medium', 'Fast', 'Very Fast']
    player_data['velocity_range'] = pd.cut(player_data['release_speed'], bins=velocity_bins, labels=velocity_labels, include_lowest=True)
    velocity_freq = player_data.groupby(['batter', 'velocity_range'])['events'].count().unstack('velocity_range', fill_value=0)
    velocity_freq.columns = [f'velocity_{col}' for col in velocity_freq.columns]

    batter_dataset = player_data.groupby('batter')[['ERA_range']].count().reset_index()
    batter_dataset = batter_dataset.merge(pitch_freq, left_on='batter', right_index=True)
    batter_dataset = batter_dataset.merge(velocity_freq, left_on='batter', right_index=True)
    batter_dataset = batter_dataset.merge(era_freq, left_on='batter', right_index=True)

    return batter_dataset

# Example usage:
start_year = 2021
end_year = 2024

player_data = get_player_data(start_year, end_year)
batter_dataset = create_batter_dataset(player_data)

# Save the batter dataset to a CSV file
batter_dataset.to_csv('batter_dataset.csv', index=False)
