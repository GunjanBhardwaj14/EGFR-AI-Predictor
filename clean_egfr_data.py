import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import warnings

warnings.filterwarnings("ignore")

print("Starting Phase 2 & 3: Data Wrangling and pIC50 Transformation...")

# 1. Load the raw dataset
df = pd.read_csv('../data/egfr_raw_data.csv')

# 2. Keep only the essential columns
keep_cols = ['molecule_chembl_id', 'canonical_smiles', 'standard_value']
df = df[keep_cols]

# 3. Clean up missing data and duplicates
df = df.dropna(subset=['canonical_smiles', 'standard_value'])
df = df.drop_duplicates(subset=['canonical_smiles'])

# IMPORTANT: Ensure standard_value is purely numeric and positive
df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
df = df.dropna(subset=['standard_value'])
df = df[df['standard_value'] > 0] # We cannot take a logarithm of zero or negative numbers!

# 4. Phase 3: Calculate pIC50 (The PhD-Level Normalization)
print("Applying advanced data normalization (Converting IC50 to pIC50)...")

def calculate_pic50(ic50):
    # Convert standard_value (in nM) to Molar (M), then apply the negative log10
    molar = ic50 * (10**-9)
    return -np.log10(molar)

# Apply the pIC50 math and create a brand new column
df['pIC50'] = df['standard_value'].apply(calculate_pic50)

# 5. Calculate chemical properties using RDKit
print("Converting 1D structures to mathematical properties (Molecular Weight & LogP)...")

def calculate_mw(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol) if mol else None

def calculate_logp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolLogP(mol) if mol else None

df['Molecular_Weight'] = df['canonical_smiles'].apply(calculate_mw)
df['LogP'] = df['canonical_smiles'].apply(calculate_logp)

# Drop any molecules that RDKit failed to process
df = df.dropna()

# 6. Save the ultimate clean dataset
df.to_csv('../data/egfr_clean_data.csv', index=False)
print(f"Success! {df.shape[0]} mathematically pristine compounds saved to data/egfr_clean_data.csv.")