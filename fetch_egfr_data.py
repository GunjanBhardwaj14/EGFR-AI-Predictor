import pandas as pd
from chembl_webresource_client.new_client import new_client
import warnings

# Suppress annoying warning messages
warnings.filterwarnings("ignore")

print("Initiating connection to the ChEMBL Database...")

# Explicitly target EGFR (Epidermal Growth Factor Receptor) - ChEMBL ID: CHEMBL203
selected_target = "CHEMBL203"
print(f"Targeting EGFR (ID: {selected_target})...")

# Extract the first 2000 lab tests to prevent server timeouts!
activity = new_client.activity
res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")[0:2000]
df = pd.DataFrame.from_dict(res)

print(f"Data Mining Complete! Extracted {df.shape[0]} lab test results.")

# Save this raw dataset to your data folder
df.to_csv('../data/egfr_raw_data.csv', index=False)
print("Saved to data/egfr_raw_data.csv. Ready for data wrangling!")