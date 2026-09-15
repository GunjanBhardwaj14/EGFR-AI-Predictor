import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
from rdkit.Chem import AllChem

# 1. Design the Website Header
st.title("EGFR Cancer Drug Predictor")
st.write("Enter a chemical's SMILES string below to let the AI predict its pIC50 potency.")

# 2. Build and Cache the AI Brain 
# @st.cache_resource tells the website to only train the AI once so it doesn't crash on reload
@st.cache_resource
def load_and_train_ai():
    df = pd.read_csv('../data/egfr_clean_data.csv')
    
    def get_fp(smiles):
        mol = Chem.MolFromSmiles(smiles)
        generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
        return list(generator.GetFingerprint(mol)) if mol else [0]*2048
        
    X = np.array(df['canonical_smiles'].apply(get_fp).tolist())
    y = df['pIC50']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = load_and_train_ai()
st.success("Advanced AI Brain Loaded & Ready!")

# 3. Create the User Input Box
# (We put Aspirin's SMILES string in there as a default example)
user_smiles = st.text_input("Enter Chemical SMILES string:", "CC(=O)Oc1ccccc1C(=O)O")

# 4. Create the Prediction Button
if st.button("Predict pIC50 Potency"):
    mol = Chem.MolFromSmiles(user_smiles)
    
    if mol: # If RDKit successfully reads the 3D shape
        # X-Ray the new drug
        generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
        fp = list(generator.GetFingerprint(mol))
        # Ask the AI to guess the score
        prediction = model.predict([fp])[0]
        # Display the result in a massive font on the website
        st.metric(label="AI Predicted pIC50", value=round(prediction, 2))
    else:
        st.error("Invalid SMILES string. Please check your spelling and try again.")