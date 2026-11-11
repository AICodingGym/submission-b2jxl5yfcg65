import sys, numpy as np, pandas as pd
d = sys.argv[1]
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
tr = pd.read_csv(f"{d}/train.csv"); te = pd.read_csv(f"{d}/test.csv"); ss = pd.read_csv(f"{d}/sample_submission.csv")
print("shapes", tr.shape, te.shape, ss.shape)
print(tr.head(3)); print(tr.dtypes.to_dict())
print("nulls train", tr.isna().sum().sum(), "test", te.isna().sum().sum())
for n, df in [("train", tr), ("test", te)]:
    g = df.groupby("breath_id")
    print(f"\n[{n}] breaths={df.breath_id.nunique()} rows/breath={g.size().unique()} id range={df.id.min()}-{df.id.max()} breath_id range={df.breath_id.min()}-{df.breath_id.max()}")
    print("R vals", sorted(df.R.unique()), "C vals", sorted(df.C.unique()))
    print("R/C constant within breath:", (g.R.nunique().max()==1) and (g.C.nunique().max()==1))
    rc = g[["R","C"]].first().value_counts().sort_index()
    print("breaths per (R,C):\n", (rc/rc.sum()*100).round(1).unstack())
    print(df[["time_step","u_in","u_out"]].describe().round(3).T)
print("\nbreath id overlap:", len(set(tr.breath_id)&set(te.breath_id)))
print("submission ids match test:", (ss.id.values==te.id.values).all(), "sample pressure unique:", ss.pressure.unique()[:5])
# inspiratory phase
insp = tr.u_out==0
print(f"\ninspiratory rows: train {insp.mean():.3%}, test {(te.u_out==0).mean():.3%}")
ni = tr[insp].groupby("breath_id").size()
print("insp steps/breath:", ni.describe().round(1).to_dict())
# is u_out monotone 0 then 1?
first1 = tr.groupby("breath_id").u_out.apply(lambda s: (s.diff().fillna(0)<0).any())
print("breaths where u_out goes 1->0:", first1.sum())
tr["step"] = tr.groupby("breath_id").cumcount()
print("time_step at last step:", tr[tr.step==79].time_step.describe().round(3).to_dict())
dt = tr.groupby("breath_id").time_step.diff().dropna()
print("dt:", dt.describe().round(4).to_dict())
# pressure
print("\npressure all:", tr.pressure.describe().round(3).to_dict())
print("pressure insp:", tr[insp].pressure.describe().round(3).to_dict())
pv = np.sort(tr.pressure.unique()); print("unique pressures:", len(pv), "min step:", np.round(np.diff(pv).min(),5), "median step:", np.round(np.median(np.diff(pv)),5))
print("pressure mean/std by (R,C) insp:\n", tr[insp].groupby(["R","C"]).pressure.agg(["mean","std"]).round(2))
print("u_in insp vs exp mean:", tr[insp].u_in.mean().round(2), tr[~insp].u_in.mean().round(2))
print("u_in==0 share in insp:", (tr[insp].u_in==0).mean().round(3), "u_in>0 share in exp:", (tr[~insp].u_in>0).mean().round(3))
# correlations insp
ti = tr[insp].copy(); g = ti.groupby("breath_id")
ti["u_in_cumsum"] = g.u_in.cumsum()
ti["u_in_lag1"] = g.u_in.shift(1)
print("corr with pressure (insp):\n", ti[["R","C","time_step","u_in","u_in_lag1","u_in_cumsum","pressure"]].corr()["pressure"].round(3))
# simple baselines on insp MAE
print("\nBaselines (insp MAE on train):")
print(" global mean:", (ti.pressure-ti.pressure.mean()).abs().mean().round(3), " global median:", (ti.pressure-ti.pressure.median()).abs().mean().round(3))
m = ti.groupby(["R","C","step"]).pressure.transform("median")
print(" median by (R,C,step):", (ti.pressure-m).abs().mean().round(3))
# negative pressures, outliers
print("negative pressure rows:", (tr.pressure<0).sum(), " max pressure rows >60:", (tr.pressure>60).sum())
