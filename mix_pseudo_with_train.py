"""
Merge original train.csv with pseudo-labeled test data
"""

import pandas as pd
import os

# Paths
TRAIN_CSV = r"data/raw/train.csv"
PSEUDO_CSV = r"pseudo/pseudo_labels_round1.csv"
OUTPUT_CSV = r"pseudo/mixed_train_round1.csv"

os.makedirs("pseudo", exist_ok=True)

print("🔍 Loading CSV files...")
df_train = pd.read_csv(TRAIN_CSV)
df_pseudo = pd.read_csv(PSEUDO_CSV)

print(f"✔ Train rows: {len(df_train)}")
print(f"✔ Pseudo-labeled rows: {len(df_pseudo)}")

# Concatenate
df_mix = pd.concat([df_train, df_pseudo], axis=0)
df_mix.to_csv(OUTPUT_CSV, index=False)

print("🎉 Created mixed training dataset!")
print(f"Saved at: {OUTPUT_CSV}")
