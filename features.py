"""Step 2: per-row features that carry breath history into each row.

A tree model sees one row at a time, so anything about the rest of the breath
(how much air went in so far, what the valve did a moment ago) must be added
as a column. Every feature is computed WITHIN a breath (groupby breath_id).
"""
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["breath_id", "id"]).copy()
    g = df.groupby("breath_id")["u_in"]

    # Lung identity: 9 (R, C) combos, as one categorical-like code too.
    df["R_C"] = df["R"] * 100 + df["C"]

    # Position in the breath.
    df["step"] = df.groupby("breath_id").cumcount()

    # History of the valve: running total ~ volume pushed into the lung.
    df["u_in_cumsum"] = g.cumsum()
    for k in (1, 2, 3):
        df[f"u_in_lag{k}"] = g.shift(k).fillna(0)
    df["u_in_lead1"] = g.shift(-1).fillna(0)
    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]

    # Whole-breath summaries (the control pattern of this breath).
    df["breath_u_in_mean"] = g.transform("mean")
    df["breath_u_in_max"] = g.transform("max")
    return df


FEATURES = [
    "R", "C", "R_C", "step", "time_step", "u_in", "u_out",
    "u_in_cumsum", "u_in_lag1", "u_in_lag2", "u_in_lag3", "u_in_lead1",
    "u_in_diff1", "breath_u_in_mean", "breath_u_in_max",
]
