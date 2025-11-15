import pandas as pd
import os

BASE_DIR = r"C:\Users\kotir\retinal_disease\data\raw"

train_csv = os.path.join(BASE_DIR, "sample_submission.csv")
pseudo_csv = os.path.join(BASE_DIR, "pseudo_submission.csv")

print("📥 Loading CSV files...")
df_train = pd.read_csv(train_csv)
df_pseudo = pd.read_csv(pseudo_csv)

print("\n===== 🔍 BASIC INFO =====")
print("Train shape :", df_train.shape)
print("Pseudo shape:", df_pseudo.shape)

# -------------------------------
# CHECK COLUMNS
# -------------------------------
print("\n===== 🔍 COLUMN COMPARISON =====")
print("Train columns :", list(df_train.columns))
print("Pseudo columns:", list(df_pseudo.columns))

if list(df_pseudo.columns) != ["id_code", "diagnosis"]:
    print("❌ Pseudo CSV columns are NOT in correct order!")
else:
    print("✅ Columns order OK")

# -------------------------------
# CHECK ID OVERLAP (should be ZERO)
# -------------------------------
print("\n===== 🔍 OVERLAPPING IDS =====")

train_ids = set(df_train["id_code"])
pseudo_ids = set(df_pseudo["id_code"])

overlap = train_ids.intersection(pseudo_ids)

print(f"• Train IDs:  {len(train_ids)}")
print(f"• Pseudo IDs: {len(pseudo_ids)}")
print(f"• Overlap:    {len(overlap)}")

if len(overlap) == 0:
    print("✅ No overlap — OK for merging")
else:
    print("❌ WARNING: Some test images already exist in train.csv")
    print("IDs:", list(overlap)[:10])  # show first 10

# -------------------------------
# CHECK LABEL VALIDITY
# -------------------------------
print("\n===== 🔍 LABEL CHECK =====")

valid_labels = {0, 1, 2, 3, 4}

invalid_train = set(df_train["diagnosis"]) - valid_labels
invalid_pseudo = set(df_pseudo["diagnosis"]) - valid_labels

print("Invalid train labels:", invalid_train)
print("Invalid pseudo labels:", invalid_pseudo)

if not invalid_pseudo:
    print("✅ All pseudo labels are valid (0–4)")

# -------------------------------
# CHECK FOR DUPLICATES
# -------------------------------
print("\n===== 🔍 DUPLICATE CHECK =====")
dup_train = df_train["id_code"].duplicated().sum()
dup_pseudo = df_pseudo["id_code"].duplicated().sum()

print("Duplicates in train :", dup_train)
print("Duplicates in pseudo:", dup_pseudo)

if dup_pseudo == 0:
    print("✅ No duplicates in pseudo")
else:
    print("❌ Pseudo CSV has duplicate test IDs!")

# -------------------------------
# LABEL DISTRIBUTION
# -------------------------------
print("\n===== 🔍 LABEL DISTRIBUTION =====")
print("\nTrain set:")
print(df_train["diagnosis"].value_counts().sort_index())

print("\nPseudo set:")
print(df_pseudo["diagnosis"].value_counts().sort_index())

print("\n🎉 Comparison finished!")
