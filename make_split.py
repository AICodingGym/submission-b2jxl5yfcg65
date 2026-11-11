"""Step 1: create a FIXED validation split of whole breaths.

Run once. Every experiment reads val_breaths.csv so all scores are comparable.
Usage: python make_split.py
"""
import numpy as np
import pandas as pd

VAL_FRACTION = 0.20
SEED = 42

train = pd.read_csv("data/train.csv", usecols=["breath_id", "R", "C"])
breaths = train.drop_duplicates("breath_id").reset_index(drop=True)

rng = np.random.default_rng(SEED)
shuffled = rng.permutation(breaths["breath_id"].to_numpy())
n_val = int(len(shuffled) * VAL_FRACTION)
val_ids = np.sort(shuffled[:n_val])

pd.DataFrame({"breath_id": val_ids}).to_csv("val_breaths.csv", index=False)

# Sanity check: does each (R, C) lung type appear in similar proportions?
breaths["split"] = np.where(breaths["breath_id"].isin(val_ids), "val", "train")
mix = (breaths.groupby("split")[["R", "C"]].value_counts(normalize=True)
       .mul(100).round(1).unstack("split"))
print(f"train breaths: {len(breaths) - n_val:,}   val breaths: {n_val:,}")
print("share of breaths per (R, C), %:\n", mix)
