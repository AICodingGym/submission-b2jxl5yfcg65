"""Assign every training breath to one of 5 folds for cross-validation.

Fold 0 = the existing fixed validation split (val_breaths.csv), so earlier runs
(v1-v6) are already valid fold-0 models. The remaining 80% of breaths are split
randomly (seed 42) into folds 1-4 of ~20% each.

Usage: python make_folds.py   -> writes folds.csv (breath_id, fold)
"""
import numpy as np
import pandas as pd

breaths = pd.read_csv("data/train.csv", usecols=["breath_id", "R", "C"]).drop_duplicates("breath_id")
val0 = set(pd.read_csv("val_breaths.csv")["breath_id"])

rest = np.sort([b for b in breaths["breath_id"] if b not in val0])
rng = np.random.default_rng(42)
rest = rng.permutation(rest)
fold_of = {b: 0 for b in val0}
for k, chunk in enumerate(np.array_split(rest, 4), start=1):
    fold_of.update({b: k for b in chunk})

breaths["fold"] = breaths["breath_id"].map(fold_of)
breaths[["breath_id", "fold"]].sort_values("breath_id").to_csv("folds.csv", index=False)

print("breaths per fold:\n", breaths["fold"].value_counts().sort_index())
print("share of breaths per (R, C) by fold, %:\n",
      (breaths.groupby("fold")[["R", "C"]].value_counts(normalize=True) * 100).round(1).unstack("fold"))
