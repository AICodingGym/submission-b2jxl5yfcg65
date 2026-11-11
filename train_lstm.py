"""Bidirectional LSTM over whole breaths (80 steps), trained with masked L1 on
inspiratory steps only. Uses the SAME fixed validation split (val_breaths.csv).

Usage (DataHub GPU):   python -u train_lstm.py --epochs 150 --max_minutes 200
Smoke test (any CPU):  python -u train_lstm.py --limit 2000 --epochs 2
Resume after a crash:  python -u train_lstm.py --resume  (same --run name)
"""
import argparse
import csv
import os
import platform
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

p = argparse.ArgumentParser()
p.add_argument("--run", default="lstm_v1")
p.add_argument("--epochs", type=int, default=150)
p.add_argument("--batch", type=int, default=512)
p.add_argument("--hidden", type=int, default=256)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--dropout", type=float, default=0.0, help="dropout between stacked LSTM layers")
p.add_argument("--wd", type=float, default=1e-5, help="AdamW weight decay")
p.add_argument("--limit", type=int, default=0, help="use only N train breaths (smoke test)")
p.add_argument("--max_minutes", type=float, default=0, help="stop training after this many minutes")
p.add_argument("--ckpt_dir", default="checkpoints")
p.add_argument("--resume", action="store_true")
p.add_argument("--eval_only", action="store_true", help="skip training; analyze best checkpoint")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--feats", choices=["v1", "v2"], default="v1", help="input feature set")
p.add_argument("--fold", type=int, default=0, choices=range(5),
               help="validation fold from folds.csv (0 = original val_breaths.csv)")
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
devname = torch.cuda.get_device_name(0) if dev.type == "cuda" else f"CPU ({platform.machine()})"
print(f"device: {devname}  args: {vars(args)}", flush=True)

# ---------------------------------------------------------------- data
# Keep u_out at index 1 and the R/C one-hots at 6:12 in every feature set
# (the mask and the error breakdown rely on those positions).
FEATS_V1 = ["u_in", "u_out", "time_step", "u_in_cumsum", "u_in_lag1", "u_in_diff1",
            "R5", "R20", "R50", "C10", "C20", "C50"]
# v2 adds physics-motivated inputs: pressure ~ R * flow + volume / C.
FEATS_V2 = FEATS_V1 + ["dt", "area", "u_in_x_R", "area_div_C", "R_val", "C_val",
                       "u_in_lag2", "u_in_lag3", "u_in_lead1"]
FEATS = {"v1": FEATS_V1, "v2": FEATS_V2}[args.feats]
BINARY = {"u_out", "R5", "R20", "R50", "C10", "C20", "C50"}
CONT = [i for i, f in enumerate(FEATS) if f not in BINARY]  # standardized columns


def to_arrays(df):
    """Rows -> (breaths, 80, features) arrays. Each breath is 80 contiguous rows."""
    df = df.sort_values(["breath_id", "id"]).copy()
    g = df.groupby("breath_id")["u_in"]
    df["u_in_cumsum"] = g.cumsum()
    df["u_in_lag1"] = g.shift(1).fillna(0)
    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    # v2 features (cheap to compute; only used if listed in FEATS)
    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0)
    df["area"] = (df["u_in"] * df["dt"]).groupby(df["breath_id"]).cumsum()
    df["u_in_x_R"] = df["u_in"] * df["R"]
    df["area_div_C"] = df["area"] / df["C"]
    df["R_val"], df["C_val"] = df["R"].astype(float), df["C"].astype(float)
    df["u_in_lag2"] = g.shift(2).fillna(0)
    df["u_in_lag3"] = g.shift(3).fillna(0)
    df["u_in_lead1"] = g.shift(-1).fillna(0)
    for r in (5, 20, 50):
        df[f"R{r}"] = (df["R"] == r).astype(np.float32)
    for c in (10, 20, 50):
        df[f"C{c}"] = (df["C"] == c).astype(np.float32)
    n = df["breath_id"].nunique()
    X = df[FEATS].to_numpy(np.float32).reshape(n, 80, len(FEATS))
    y = df["pressure"].to_numpy(np.float32).reshape(n, 80) if "pressure" in df else None
    ids = df["id"].to_numpy().reshape(n, 80)
    bids = df["breath_id"].to_numpy().reshape(n, 80)[:, 0]
    return X, y, ids, bids


t0 = time.time()
X, y, _, bids = to_arrays(pd.read_csv("data/train.csv"))
Xte, _, te_ids, _ = to_arrays(pd.read_csv("data/test.csv"))
if args.fold == 0:  # fold 0 is the original fixed split -> earlier runs stay comparable
    val_ids = pd.read_csv("val_breaths.csv")["breath_id"].to_numpy()
else:
    folds = pd.read_csv("folds.csv")
    val_ids = folds.loc[folds["fold"] == args.fold, "breath_id"].to_numpy()
is_val = np.isin(bids, val_ids)
Xtr, ytr, Xva, yva, bva = X[~is_val], y[~is_val], X[is_val], y[is_val], bids[is_val]
if args.limit:
    Xtr, ytr = Xtr[: args.limit], ytr[: args.limit]
    Xva, yva, bva = Xva[: args.limit // 4], yva[: args.limit // 4], bva[: args.limit // 4]

# Standardize continuous features with TRAIN-split statistics only (no peeking at val/test).
mu = Xtr[:, :, CONT].reshape(-1, len(CONT)).mean(0)
sd = Xtr[:, :, CONT].reshape(-1, len(CONT)).std(0) + 1e-6
for A in (Xtr, Xva, Xte):
    A[:, :, CONT] = (A[:, :, CONT] - mu) / sd
print(f"data ready {time.time() - t0:.1f}s  train {len(Xtr):,}  val {len(Xva):,}  "
      f"test {len(Xte):,} breaths", flush=True)

# Whole dataset fits in GPU memory (~200 MB), so keep it there.
T = lambda a: torch.tensor(a, device=dev)
Xtr_t, ytr_t, Xva_t, yva_t, Xte_t = T(Xtr), T(ytr), T(Xva), T(yva), T(Xte)
mtr_t = Xtr_t[:, :, 1] == 0  # u_out column is not standardized -> mask = inspiratory
mva_t = Xva_t[:, :, 1] == 0


# ---------------------------------------------------------------- model
class BreathLSTM(nn.Module):
    def __init__(self, n_feat, hidden, layers, dropout=0.0):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_feat, hidden), nn.GELU())
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, dropout=dropout,
                            batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(),
                                  nn.Linear(hidden, 1))

    def forward(self, x):                # x: (batch, 80, n_feat)
        z, _ = self.lstm(self.inp(x))    # z: (batch, 80, 2*hidden) forward+backward memory
        return self.head(z).squeeze(-1)  # (batch, 80) one pressure per step


def masked_l1(pred, target, mask):
    """MAE over inspiratory steps only -- the leaderboard metric."""
    return (pred - target).abs()[mask].mean()


@torch.no_grad()
def predict(Xt, bs=2048):
    model.eval()
    return torch.cat([model(Xt[i:i + bs]) for i in range(0, len(Xt), bs)])


model = BreathLSTM(len(FEATS), args.hidden, args.layers, args.dropout).to(dev)
# Start predictions near the typical pressure instead of 0, so early epochs aren't wasted
# just shifting the output up by ~16 cmH2O (L1 gradients are small and constant).
with torch.no_grad():
    model.head[-1].bias.fill_(float(np.median(ytr[Xtr[:, :, 1] == 0])))
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

os.makedirs(args.ckpt_dir, exist_ok=True)
last_path = os.path.join(args.ckpt_dir, f"{args.run}_last.pt")
best_path = os.path.join(args.ckpt_dir, f"{args.run}_best.pt")
start_epoch, best_mae, train_secs = 0, float("inf"), 0.0
if args.resume and os.path.exists(last_path):
    ck = torch.load(last_path, map_location=dev)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    start_epoch, best_mae, train_secs = ck["epoch"] + 1, ck["best_mae"], ck["train_secs"]
    print(f"resumed from epoch {start_epoch}, best val MAE {best_mae:.4f}", flush=True)

# ---------------------------------------------------------------- train
t_start = time.time()
for epoch in range(start_epoch, 0 if args.eval_only else args.epochs):
    te = time.time()
    model.train()
    perm = torch.randperm(len(Xtr_t), device=dev)
    tot, nb = 0.0, 0
    for i in range(0, len(perm), args.batch):
        b = perm[i:i + args.batch]
        loss = masked_l1(model(Xtr_t[b]), ytr_t[b], mtr_t[b])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot, nb = tot + loss.item(), nb + 1
    sched.step()
    val_mae = masked_l1(predict(Xva_t), yva_t, mva_t).item()
    train_secs += time.time() - te
    if val_mae < best_mae:
        best_mae = val_mae
        torch.save({"model": model.state_dict(), "mu": mu.tolist(), "sd": sd.tolist(), "args": vars(args)}, best_path)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "epoch": epoch, "best_mae": best_mae, "train_secs": train_secs}, last_path)
    print(f"epoch {epoch + 1:3d}/{args.epochs}  train L1 {tot / nb:.4f}  val MAE {val_mae:.4f}  "
          f"best {best_mae:.4f}  lr {sched.get_last_lr()[0]:.2e}  {time.time() - te:.1f}s", flush=True)
    if args.max_minutes and (time.time() - t_start) / 60 > args.max_minutes:
        print(f"time limit {args.max_minutes} min reached; stopping (resume with --resume)", flush=True)
        break

# ---------------------------------------------------------------- predict + log
model.load_state_dict(torch.load(best_path, map_location=dev)["model"])

# Where are the validation errors? (guides the next experiment)
va_pred = predict(Xva_t)
np.savez(f"val_pred_{args.run}.npz", pred=va_pred.cpu().numpy(), y=yva, mask=mva_t.cpu().numpy(),
         bids=bva, fold=args.fold)
err = (va_pred - yva_t).abs().cpu().numpy()
insp = mva_t.cpu().numpy()
R = np.array([5, 20, 50])[Xva[:, 0, 6:9].argmax(1)]
C = np.array([10, 20, 50])[Xva[:, 0, 9:12].argmax(1)]
e = pd.DataFrame({"R": np.repeat(R, 80), "C": np.repeat(C, 80), "step": np.tile(np.arange(80), len(R)),
                  "breath": np.repeat(np.arange(len(R)), 80), "err": err.ravel()})[insp.ravel()]
if args.eval_only:
    best_mae = float(e.err.mean())
print(f"\nval insp MAE (best ckpt) {e.err.mean():.4f}")
print("by lung (R rows, C cols):\n", e.groupby(["R", "C"]).err.mean().unstack().round(3))
print("by step bucket:\n", e.groupby(pd.cut(e.step, [-1, 4, 9, 19, 31]), observed=True).err.mean().round(3))
per_breath = e.groupby("breath").err.mean()
worst = per_breath.nlargest(len(per_breath) // 100)
print(f"per-breath MAE: median {per_breath.median():.3f}  p90 {per_breath.quantile(.9):.3f}  "
      f"max {per_breath.max():.3f};  worst 1% of breaths = "
      f"{e[e.breath.isin(worst.index)].err.sum() / e.err.sum():.1%} of total error")

pred_te = predict(Xte_t).cpu().numpy()
sub = pd.DataFrame({"id": te_ids.ravel(), "pressure": pred_te.ravel()}).sort_values("id")
assert np.isfinite(sub["pressure"]).all()
sub_path = f"submission_{args.run}.csv"
sub.to_csv(sub_path, index=False)

if not args.limit and not args.eval_only:  # keep smoke tests / re-analysis out of the log
    new = not os.path.exists("results.csv")
    with open("results.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "run", "val_insp_mae", "fit_seconds", "device", "notes"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), args.run, f"{best_mae:.4f}",
                    f"{train_secs:.0f}", devname, str(vars(args))])
print(f"best val insp MAE {best_mae:.4f}  train time {train_secs / 60:.1f} min  "
      f"wrote {sub_path} ({len(sub):,} rows)", flush=True)
