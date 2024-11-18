from nba_api.stats.endpoints import leaguegamefinder, scoreboardv2
from nba_api.stats.static import teams
import pandas as pd
import joblib
import datetime

# Load the trained model
print("Loading trained model...")
model_path = "/Users/colintroy/Documents/python/nba_predictor/nba_game_predictor.pkl"  # Update with your model file path
model = joblib.load(model_path)
print("Model loaded successfully!")

# Load team dictionary for team names
nba_teams = teams.get_teams()
team_dict = {team['id']: team['full_name'] for team in nba_teams}


def get_games_on_date(date_str):
    """Fetch and list NBA games for the given date."""
    date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    print(f"Fetching games for {date_str}...")
    scoreboard = scoreboardv2.ScoreboardV2(game_date=date_obj.strftime('%m/%d/%Y'))
    games_data = scoreboard.game_header.get_data_frame()

    if games_data.empty:
        print(f"No games scheduled for {date_str}.")
        return []

    print(f"\nGames on {date_str}:")
    games_list = []
    for idx, game in games_data.iterrows():
        home_team_id = game['HOME_TEAM_ID']
        visitor_team_id = game['VISITOR_TEAM_ID']
        home_team_name = team_dict.get(home_team_id, "Unknown Team")
        visitor_team_name = team_dict.get(visitor_team_id, "Unknown Team")
        print(f"{idx + 1}: {home_team_name} vs. {visitor_team_name}")
        games_list.append({
            'GAME_ID': game['GAME_ID'],
            'HOME_TEAM_ID': home_team_id,
            'VISITOR_TEAM_ID': visitor_team_id
        })
    return games_list


def get_team_last_n_games_stats(team_id, num_games=10):
    """Fetch and aggregate stats for the last n games of a team."""
    gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id)
    games = gamefinder.get_data_frames()[0]
    recent_games = games.sort_values(by='GAME_DATE').tail(num_games)

    # Define stats to aggregate
    stats_to_aggregate = {
        'PTS': 'mean', 'REB': 'mean', 'AST': 'mean', 'FGM': 'mean', 'FGA': 'mean',
        'FG_PCT': 'mean', 'FG3M': 'mean', 'STL': 'mean', 'BLK': 'mean', 'TOV': 'mean'
    }
    available_stats = {key: val for key, val in stats_to_aggregate.items() if key in recent_games.columns}
    team_stats = recent_games.agg(available_stats).to_dict()

    for stat in stats_to_aggregate.keys():
        if stat not in team_stats:
            team_stats[stat] = 0  # Default to 0 if stat is unavailable
    return team_stats


def predict_selected_game_winner(home_team_id, visitor_team_id):
    """Predict the winner of a game using the model."""
    print("\nFetching stats for the home team...")
    home_team_stats = get_team_last_n_games_stats(home_team_id, num_games=10)
    print("Home Team Stats:", home_team_stats)

    print("\nFetching stats for the visitor team...")
    visitor_team_stats = get_team_last_n_games_stats(visitor_team_id, num_games=10)
    print("Visitor Team Stats:", visitor_team_stats)

    # Prepare data for model input, ensuring feature names match training
    game_data = {
        'PTS_rolling_A': home_team_stats['PTS'], 'REB_rolling_A': home_team_stats['REB'], 'AST_rolling_A': home_team_stats['AST'],
        'FGM_rolling_A': home_team_stats['FGM'], 'FGA_rolling_A': home_team_stats['FGA'], 'FG_PCT_rolling_A': home_team_stats['FG_PCT'],
        'FG3M_rolling_A': home_team_stats['FG3M'], 'STL_rolling_A': home_team_stats['STL'], 'BLK_rolling_A': home_team_stats['BLK'], 'TO_rolling_A': home_team_stats['TOV'],
        'PTS_rolling_B': visitor_team_stats['PTS'], 'REB_rolling_B': visitor_team_stats['REB'], 'AST_rolling_B': visitor_team_stats['AST'],
        'FGM_rolling_B': visitor_team_stats['FGM'], 'FGA_rolling_B': visitor_team_stats['FGA'], 'FG_PCT_rolling_B': visitor_team_stats['FG_PCT'],
        'FG3M_rolling_B': visitor_team_stats['FG3M'], 'STL_rolling_B': visitor_team_stats['STL'], 'BLK_rolling_B': visitor_team_stats['BLK'], 'TO_rolling_B': visitor_team_stats['TOV']
    }

    # Convert to DataFrame for model compatibility
    game_df = pd.DataFrame([game_data])

    # Make prediction
    prediction = model.predict(game_df)[0]
    probabilities = model.predict_proba(game_df)[0]

    # Determine the winning team and confidence
    home_team_name = team_dict.get(home_team_id, "Unknown Home Team")
    visitor_team_name = team_dict.get(visitor_team_id, "Unknown Visitor Team")
    winning_team = home_team_name if prediction == 1 else visitor_team_name
    confidence = probabilities[prediction] * 100

    return f"The predicted winner is {winning_team} with {confidence:.2f}% confidence."



def main():
    date_str = input("Enter a date (YYYY-MM-DD) to see games: ")
    games_list = get_games_on_date(date_str)
    if not games_list:
        return

    game_idx = int(input("Select a game by number: ")) - 1
    selected_game = games_list[game_idx]

    prediction = predict_selected_game_winner(
        selected_game['HOME_TEAM_ID'], selected_game['VISITOR_TEAM_ID']
    )
    print(prediction)


if __name__ == "__main__":
    main()
