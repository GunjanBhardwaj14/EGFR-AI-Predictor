import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

print("Phase 4: Initializing AI Training Sequence...")

# 1. Load the clean, mathematical dataset
df = pd.read_csv('../data/egfr_clean_data.csv')

# 2. Define the Inputs (X) and the Target (y)
X = df[['Molecular_Weight', 'LogP']] # The chemical features
y = df['pIC50']                      # The actual drug potency

# 3. Split the data into Training and Testing sets
# We hide 20% of the data (test_size=0.2) to quiz the AI later
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training AI on {X_train.shape[0]} compounds. Quizzing it on {X_test.shape[0]} unseen compounds...")

# 4. Build and Train the Random Forest Brain
model = RandomForestRegressor(n_estimators=100, random_state=42)
# X_train and Y_train give AI for learning which is 80% of the data, while X_test and Y_test are hidden for later evaluation
model.fit(X_train, y_train) 

# 5. Quiz the AI on the hidden test set
# X_test is hown to AI and it predicts the pIC50 values, based on what it learned with train values.
predictions = model.predict(X_test)

# 6. Grade the AI
r2 = r2_score(y_test, predictions)
print(f"AI Training Complete! Model Accuracy Score (R2): {r2:.2f}")

# 7. Draw the Performance Graph
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.5, color='blue')
plt.title("AI Predictions vs Actual Lab Results")
plt.xlabel("Actual pIC50 (Real Lab Test)")
plt.ylabel("Predicted pIC50 (AI Guess)")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')

# Save the graph to your results folder
plt.savefig('../results/prediction_graph.png')
print("Performance graph saved to results/prediction_graph.png. Check it out!")