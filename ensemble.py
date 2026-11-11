"""Average several finished runs. Scores each run and the average on the fixed
validation breaths (inspiratory MAE), then writes the averaged test submission.

Needs val_pred_<run>.npz and submission_<run>.csv for every run
(train_lstm.py writes both; --eval_only regenerates them from a checkpoint).

Usage: python ensemble.py lstm_v1 lstm_v2_dropout lstm_v3_seed7 [--out submission_ens.csv]
"""
import argparse

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("runs", nargs="+")
p.add_argument("--out", default="submission_ens.csv")
args = p.parse_args()

vals = {r: np.load(f"val_pred_{r}.npz") for r in args.runs}
y, mask = vals[args.runs[0]]["y"], vals[args.runs[0]]["mask"]
for r, v in vals.items():  # every run must be scored on the same breaths
    assert np.array_equal(v["y"], y), f"{r} used a different validation set"
    print(f"{r:28s} val insp MAE {np.abs(v['pred'] - y)[mask].mean():.4f}")

avg = np.mean([v["pred"] for v in vals.values()], axis=0)
print(f"{'AVERAGE of ' + str(len(vals)) + ' runs':28s} val insp MAE {np.abs(avg - y)[mask].mean():.4f}")

subs = [pd.read_csv(f"submission_{r}.csv") for r in args.runs]
for s in subs[1:]:
    assert (s["id"].values == subs[0]["id"].values).all()
out = pd.DataFrame({"id": subs[0]["id"], "pressure": np.mean([s["pressure"] for s in subs], axis=0)})
assert len(out) == 603600 and np.isfinite(out["pressure"]).all()
out.to_csv(args.out, index=False)
print(f"wrote {args.out}")
