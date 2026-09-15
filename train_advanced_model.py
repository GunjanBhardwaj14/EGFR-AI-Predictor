import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings

warnings.filterwarnings("ignore")

print("Phase 4.1: Initializing Advanced AI (2,048 Morgan Fingerprint Clues)...")

# 1. Load the clean dataset
df = pd.read_csv('../data/egfr_clean_data.csv')

# 2. The Upgrade: Generate 2,048 Clues per molecule
print("X-Raying chemicals to generate 3D fingerprints...")

def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # This creates a list of 2048 1s and 0s representing the exact chemical structure
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

# Apply the x-ray to our chemicals and set them as the new Clues (X)
X = np.array(df['canonical_smiles'].apply(get_fingerprint).tolist())
y = df['pIC50']

# 3. Split into Study Guide and Final Exam
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Advanced Brain
print(f"Training AI on {X_train.shape[0]} compounds. This might take a few extra seconds...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. The Final Exam
predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
print(f"Advanced AI Training Complete! New Accuracy Score (R2): {r2:.2f}")

# 6. Draw the New Graph
plt.figure(figsize=(8, 6))
# We will make these dots purple to distinguish our advanced model!
plt.scatter(y_test, predictions, alpha=0.5, color='purple')
plt.title("Advanced AI Predictions vs Actual Lab Results")
plt.xlabel("Actual pIC50 (Real Lab Test)")
plt.ylabel("Predicted pIC50 (Advanced AI Guess)")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')

plt.savefig('../results/advanced_prediction_graph.png')
print("Graph saved to results/advanced_prediction_graph.png. Check out the difference!")

import pickle

# Save the trained model to the results folder
with open('../results/model.pkl', 'wb') as f:
    pickle.dump(model, f)