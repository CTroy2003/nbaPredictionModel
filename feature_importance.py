import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Paths UPDATE UPDATE UPDATE with your respective paths
model_path = "/Users/colintroy/Documents/python/nba_predictor/nba_game_predictor.pkl"

# Define the features used in the model
features = [
    'PTS_A', 'REB_A', 'AST_A', 'FGM_A', 'FGA_A', 'FG_PCT_A', 'FG3M_A', 'STL_A', 'BLK_A', 'TO_A',
    'PTS_B', 'REB_B', 'AST_B', 'FGM_B', 'FGA_B', 'FG_PCT_B', 'FG3M_B', 'STL_B', 'BLK_B', 'TO_B'
]

# Load the model
print("Loading the trained model...")
model = joblib.load(model_path)

# Load accuracy information if it was saved with the model
try:
    accuracy = model.accuracy_
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
except AttributeError:
    print("\nModel accuracy information not available. Ensure you calculate and save it during training.")

# Get feature importances
print("Calculating feature importances...")
importances = model.feature_importances_
feature_importances = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Display the feature importances
print("\nFeature Importances:")
print(feature_importances)

# Plot feature importances
def plot_feature_importances(feature_importances):
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances['Feature'], feature_importances['Importance'], align='center')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance in Random Forest Model')
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
    plt.tight_layout()
    plt.show()

plot_feature_importances(feature_importances)

