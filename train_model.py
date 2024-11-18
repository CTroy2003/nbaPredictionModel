import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Paths UPDATE UPDATE UPDATE with your respective paths
file_path = "/Users/colintroy/Documents/python/nba_predictor/games_details.csv" 
model_path = "/Users/colintroy/Documents/python/nba_predictor/nba_game_predictor.pkl"

def calculate_past_performance_stats(data, num_games=10):
    """
    Calculate rolling stats for each team based on the last `num_games`.
    """
    print(f"Calculating rolling stats for the past {num_games} games...")
    # Sort data by GAME_ID to simulate chronological order
    data = data.sort_values(['TEAM_ID', 'GAME_ID'])

    # Calculate rolling averages for each team
    rolling_stats = data.groupby('TEAM_ID').rolling(num_games, on='GAME_ID').agg({
        'PTS': 'mean',
        'REB': 'mean',
        'AST': 'mean',
        'FGM': 'mean',
        'FGA': 'mean',
        'FG_PCT': 'mean',
        'FG3M': 'mean',
        'STL': 'mean',
        'BLK': 'mean',
        'TO': 'mean'
    }).reset_index(drop=True)

    # Merge rolling stats back with the original data
    return pd.concat([data.reset_index(drop=True), rolling_stats.add_suffix('_rolling')], axis=1)

def prepare_training_data(data):
    """
    Prepare training data by merging rolling stats for both teams into a single row per game.
    """
    # Calculate rolling stats
    rolling_stats = calculate_past_performance_stats(data)
    rolling_stats = rolling_stats.dropna()  # Drop rows without enough past games for rolling stats

    # Separate Team A and Team B data for each game
    team_a = rolling_stats.groupby('GAME_ID').nth(0).reset_index()
    team_b = rolling_stats.groupby('GAME_ID').nth(1).reset_index()

    # Add suffixes to distinguish stats for Team A and Team B
    team_a = team_a.add_suffix('_A').rename(columns={'GAME_ID_A': 'GAME_ID'})
    team_b = team_b.add_suffix('_B').rename(columns={'GAME_ID_B': 'GAME_ID'})

    # Merge stats for Team A and Team B into a single row per game
    return pd.merge(team_a, team_b, on='GAME_ID')

def main():
    # Load the dataset
    print("Loading dataset...")
    raw_data = pd.read_csv(file_path, low_memory=False)

    # Drop unnecessary columns
    columns_to_drop = ['NICKNAME', 'COMMENT', 'START_POSITION', 'MIN']
    raw_data = raw_data.drop(columns=columns_to_drop)

    # Prepare training data
    print("Preparing training data...")
    game_data = prepare_training_data(raw_data)

    # Define features and target
    features = [
        'PTS_rolling_A', 'REB_rolling_A', 'AST_rolling_A', 'FGM_rolling_A', 'FGA_rolling_A',
        'FG_PCT_rolling_A', 'FG3M_rolling_A', 'STL_rolling_A', 'BLK_rolling_A', 'TO_rolling_A',
        'PTS_rolling_B', 'REB_rolling_B', 'AST_rolling_B', 'FGM_rolling_B', 'FGA_rolling_B',
        'FG_PCT_rolling_B', 'FG3M_rolling_B', 'STL_rolling_B', 'BLK_rolling_B', 'TO_rolling_B'
    ]
    game_data['Winner'] = (game_data['PTS_A'] > game_data['PTS_B']).astype(int)
    X = game_data[features]
    y = game_data['Winner']

    # Split data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=200)

    # Train the model
    print("Training the model...")
    model = RandomForestClassifier(random_state=200)
    model.fit(X_train, y_train)

    # Evaluate model performance
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Save the model
    print("Saving the trained model...")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}.")

if __name__ == "__main__":
    main()
