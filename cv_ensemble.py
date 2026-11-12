"""Combine runs trained on different cross-validation folds.

- Within each fold: average that fold's runs (e.g. v1..v6 all hold out fold 0).
- Out-of-fold (OOF) score: each held-out breath is predicted only by models that
  never trained on it; MAE over all covered breaths, inspiratory steps only.
- Test submission: mean over folds of each fold's average (equal weight per fold
  by default, so a fold with many runs doesn't dominate; --weight model to change).

Needs val_pred_<run>.npz and submission_<run>.csv for every run.
Usage: python cv_ensemble.py lstm_v1 lstm_v4_feat2 lstm_f1 lstm_f2 ... [--out submission_cv.csv]
"""
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("runs", nargs="+")
p.add_argument("--out", default="submission_cv.csv")
p.add_argument("--weight", choices=["fold", "model"], default="fold")
args = p.parse_args()

fold0_ids = np.sort(pd.read_csv("val_breaths.csv")["breath_id"].to_numpy())
by_fold = defaultdict(list)
full_runs = []  # --fold -1 runs trained on every breath: test predictions only, no OOF score
for r in args.runs:
    z = np.load(f"val_pred_{r}.npz")
    fold = int(z["fold"]) if "fold" in z else 0          # runs before --fold existed = fold 0
    if fold == -1:
        full_runs.append(r)
        print(f"all-data {r:28s} (trained on every breath: test predictions only)")
        continue
    bids = z["bids"] if "bids" in z else fold0_ids
    mae = np.abs(z["pred"] - z["y"])[z["mask"]].mean()
    print(f"fold {fold}  {r:28s} val insp MAE {mae:.4f}")
    by_fold[fold].append((r, z["pred"], z["y"], z["mask"], bids))

abs_sum, n_sum = 0.0, 0
for fold in sorted(by_fold):
    runs = by_fold[fold]
    bids = runs[0][4]
    for r, *_rest, b in runs[1:]:
        assert np.array_equal(b, bids), f"{r}: different breaths than others in fold {fold}"
    avg = np.mean([x[1] for x in runs], axis=0)
    err = np.abs(avg - runs[0][2])[runs[0][3]]
    abs_sum, n_sum = abs_sum + err.sum(), n_sum + err.size
    print(f"fold {fold}  average of {len(runs)} run(s){'':10s} val insp MAE {err.mean():.4f}")
covered = sum(len(by_fold[f][0][4]) for f in by_fold)
print(f"OOF insp MAE over {len(by_fold)} fold(s), {covered:,} breaths: {abs_sum / n_sum:.4f}")

# Test predictions.
sub = {r: pd.read_csv(f"submission_{r}.csv") for r in args.runs}
ids = sub[args.runs[0]]["id"].values
for r, s in sub.items():
    assert (s["id"].values == ids).all(), r
if args.weight == "fold":
    per_fold = [np.mean([sub[x[0]]["pressure"].values for x in by_fold[f]], axis=0) for f in by_fold]
    per_fold += [sub[r]["pressure"].values for r in full_runs]  # each all-data run counts as one group
    pred = np.mean(per_fold, axis=0)
else:
    pred = np.mean([s["pressure"].values for s in sub.values()], axis=0)
out = pd.DataFrame({"id": ids, "pressure": pred})
assert len(out) == 603600 and np.isfinite(out["pressure"]).all()
out.to_csv(args.out, index=False)
print(f"wrote {args.out} (test weighting: per {args.weight})")
