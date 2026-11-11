"""Step 3+4: train a gradient-boosted tree model, score it on the fixed
validation breaths (inspiratory MAE), log the run, and write submission.csv.

Usage: python train_hgb.py
"""
import csv
import os
import platform
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features import FEATURES, add_features

RUN_NAME = "hgb_v1_basic_feats"
PARAMS = dict(loss="absolute_error", learning_rate=0.1, max_iter=500,
              max_leaf_nodes=63, early_stopping=False, random_state=42)


def insp_mae(y_true, y_pred, u_out):
    """The leaderboard metric: MAE on inspiratory rows only (u_out == 0)."""
    m = u_out == 0
    return float(np.mean(np.abs(y_true[m] - y_pred[m])))


t0 = time.time()
train = add_features(pd.read_csv("data/train.csv"))
test = add_features(pd.read_csv("data/test.csv"))
val_ids = set(pd.read_csv("val_breaths.csv")["breath_id"])
is_val = train["breath_id"].isin(val_ids)
tr, va = train[~is_val], train[is_val]
print(f"loaded + features: {time.time() - t0:.1f}s  "
      f"train rows {len(tr):,}  val rows {len(va):,}")

# --- Reference: median pressure per (R, C, step), learned on train breaths only.
ref = tr[tr.u_out == 0].groupby(["R", "C", "step"])["pressure"].median().rename("ref")
va_ref = va.join(ref, on=["R", "C", "step"])["ref"].fillna(tr.pressure.median())
ref_mae = insp_mae(va.pressure.to_numpy(), va_ref.to_numpy(), va.u_out.to_numpy())
print(f"[reference] median by (R,C,step)  val insp MAE = {ref_mae:.4f}")

# --- Model: fit on inspiratory rows only (the only rows that are scored).
fit_rows = tr[tr.u_out == 0]
model = HistGradientBoostingRegressor(**PARAMS)
t1 = time.time()
model.fit(fit_rows[FEATURES], fit_rows["pressure"])
fit_s = time.time() - t1

va_pred = model.predict(va[FEATURES])
val_mae = insp_mae(va.pressure.to_numpy(), va_pred, va.u_out.to_numpy())
print(f"[{RUN_NAME}] val insp MAE = {val_mae:.4f}   fit time {fit_s:.0f}s")

# --- Where are the errors? (guides the next experiment)
va = va.assign(abs_err=np.abs(va.pressure - va_pred))
vi = va[va.u_out == 0]
print("\nval insp MAE by lung (R rows, C cols):\n",
      vi.groupby(["R", "C"]).abs_err.mean().unstack().round(3))
print("\nval insp MAE by step bucket:\n",
      vi.groupby(pd.cut(vi.step, [-1, 4, 9, 19, 31])).abs_err.mean().round(3))

# --- Log the run so every experiment is comparable.
device = f"CPU ({platform.machine()}, {os.cpu_count()} cores)"
log_exists = os.path.exists("results.csv")
with open("results.csv", "a", newline="") as f:
    w = csv.writer(f)
    if not log_exists:
        w.writerow(["timestamp", "run", "val_insp_mae", "fit_seconds", "device", "notes"])
    stamp = datetime.now().isoformat(timespec="seconds")
    w.writerow([stamp, "ref_median_RC_step", f"{ref_mae:.4f}", 0, device, "lookup table baseline"])
    w.writerow([stamp, RUN_NAME, f"{val_mae:.4f}", f"{fit_s:.0f}", device, str(PARAMS)])

# --- Submission: predict every test row (expiratory rows are unscored but must be finite).
test["pressure"] = model.predict(test[FEATURES])
sub = test.sort_values("id")[["id", "pressure"]]
assert len(sub) == 603600 and np.isfinite(sub.pressure).all()
sub.to_csv("submission.csv", index=False)
print(f"\nwrote submission.csv ({len(sub):,} rows). total {time.time() - t0:.0f}s")
