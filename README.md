# nbaPredictionModel

File List:

nba_predictor/ ├── requirements.txt # MUST INSTALL THE LIBRARIES HERE TO RUN LOCALLY ├── games_details.csv # Training data ├── train_model.py # UPDATE FILE AND MODEL PATHS TO RUN LOCALLY ├── feature_importance.py # UPDATE MODEL PATH TO RUN LOCALLY ├── predict_games.py # UPDATE MODEL PATH TO RUN LOCALLY └── results.txt # Storing local results

TO RUN LOCALLY FOLLOW THESE STEPS:

  1. Download games_details.csv from ()
  2. Adjust necessary file and model paths in the files listed above. There will be a comment indicating where adjustments are needed.
  3. Run train_model.py
  4. (Optional) Run feature_importance.py to see what data columns our model values when making its prediction.
  5. Run predict_games.py, enter a date in the format(2024-XX-XX) and select the matchup you want to predict.

File summaries:

train_model.py - Cleans data from games_details.csv, uses rolling averages from past games when training and testing using a logistic regression model. It will output the result of the model and will save the model in nba_game_predictor.pkl to be used later by feature_importance.py and predict_games.py.

feature_importance.py - Analyzes the trained model, listing each feature and its importance in the model.

predict_games.py - This program leverages the nba_api to gather the nba games for the date the user provides. The user can then select any of the games for that day, the program pulls up the current rolling averages of the team and outputs the likely winner as well as the models confidence in its prediction.

  
            
                
                                 
