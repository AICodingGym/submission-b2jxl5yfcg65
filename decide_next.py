"""Pick the next experiments once the two single-model tests finish, then restart the queue.

Rule (decided in advance, so it runs even if nobody is watching):
  - Read the fold-0 validation MAE of test 1 (lower learning rate) and test 2 (weight decay).
  - If the better one is <= THRESHOLD (beats every single model so far, 0.1914 best),
    train that setting on folds 1-4 -> 4 better fold models for the blend.
  - Otherwise, neither setting clearly helps: train 3 models on ALL breaths (--fold -1)
    with the usual settings, so each model sees 25% more data.
Writes the choice to queue_history.txt and starts run_queue.sh again.

Usage (in the pod, in the background):  python -u decide_next.py > decide_next.out 2>&1
"""
import os
import subprocess
import time

import numpy as np

HOME = os.environ.get("VPP_DIR", os.path.expanduser("~/ventilator-pressure-prediction"))
os.chdir(HOME)
THRESHOLD = 0.190
TESTS = {  # run name -> extra flags that define the setting
    "lstm_t1_lr5e4": "--lr 5e-4",
    "lstm_t2_wd5e2": "--wd 5e-2",
}
LAST_TEST = "lstm_t2_wd5e2"


def history():
    with open("queue_history.txt") as f:
        return [l.strip() for l in f if l.strip()]


def log(msg):
    with open("queue_history.txt", "a") as f:
        f.write(f"{time.strftime('%F %T')} DECIDE {msg}\n")


# 1) Wait until the last test has ended and the queue runner has exited.
while True:
    h = history()
    if any(f"END {LAST_TEST}" in l for l in h) and h[-1].endswith("QUEUE EMPTY"):
        break
    time.sleep(60)


# 2) Score both tests on their fold-0 validation predictions.
def mae(run):
    try:
        z = np.load(f"val_pred_{run}.npz")
        return float(np.abs(z["pred"] - z["y"])[z["mask"]].mean())
    except (OSError, KeyError):
        return float("inf")


scores = {r: mae(r) for r in TESTS}
best = min(scores, key=scores.get)
log("scores " + ", ".join(f"{r}={s:.4f}" for r, s in scores.items()))

# 3) Choose the next runs.
base = "--feats v2 --seed 42 --epochs 200 --max_minutes 115"
if scores[best] <= THRESHOLD:
    tag = best.split("_", 2)[2]  # e.g. lr5e4
    lines = [f"--run lstm_f{k}_{tag} --fold {k} {base} {TESTS[best]}" for k in (1, 2, 3, 4)]
    log(f"winner {best} ({scores[best]:.4f} <= {THRESHOLD}) -> folds 1-4 with '{TESTS[best]}'")
else:
    lines = [f"--run lstm_full_s{s} --fold -1 --feats v2 --seed {s} --epochs 200 --max_minutes 140"
             for s in (1, 2, 3)]
    log(f"no winner (best {best} {scores[best]:.4f} > {THRESHOLD}) -> 3 all-data runs")

with open("queue.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

# 4) Restart the queue runner, detached from this script.
subprocess.Popen(["sh", "run_queue.sh"], stdout=open("queue_runner.out", "a"),
                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
log("queue restarted")
