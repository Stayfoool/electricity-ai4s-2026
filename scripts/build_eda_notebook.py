"""Generate reports/eda.ipynb programmatically.

Sections:
  A Forecast quality
  B Price structure
  C Input -> price
  D Train-test shift + spring festival
  E Oracle dispatch difficulty
  F Conclusions
"""

# ruff: noqa: E501
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def co(s): cells.append(nbf.v4.new_code_cell(s))

md("""# 蒙西电价 EDA — 2025 全年

**目的**：在加新特征/换模型前，先看清数据规律。

**数据**:
- `boundary` (35040 行 = 365×96)：7 channel × (实际值, 预测值)
- `node_price` (34923 行)：实际电价 `A`
- 测试集 (`to_sais_new/test/test_in_feature_ori.csv`) 只有预测值

**净负荷**：`net_load = 系统负荷 − 风光 − 水电 − 非市场化 − 联络线`

**章节**：A 预测质量 / B 价格结构 / C 输入→价格 / D 分布偏移+春节 / E 调度难度 / F 结论""")

co(r"""import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path

plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = True
plt.rcParams["font.sans-serif"] = ["Heiti TC","Arial Unicode MS","PingFang SC","DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path("/Users/yw/ai/electricity")
OUT  = ROOT / "reports/eda"; OUT.mkdir(parents=True, exist_ok=True)

bnd  = pd.read_csv(ROOT/"eletricmaterial/to_sais_new/train/mengxi_boundary_anon_filtered.csv", parse_dates=["times"])
prc  = pd.read_csv(ROOT/"eletricmaterial/to_sais_new/train/mengxi_node_price_selected.csv",   parse_dates=["times"])
test = pd.read_csv(ROOT/"eletricmaterial/to_sais_new/test/test_in_feature_ori.csv",            parse_dates=["times"])

CH  = ["系统负荷","风光总加","联络线","风电","光伏","水电","非市场化机组"]
ACT = [c+"实际值" for c in CH]
FCT = [c+"预测值" for c in CH]

print("boundary:", bnd.shape, "range:", bnd.times.min(), "~", bnd.times.max())
print("price   :", prc.shape, "range:", prc.times.min(), "~", prc.times.max())
print("test    :", test.shape,"range:", test.times.min(),"~", test.times.max())

df = bnd.merge(prc, on="times", how="left")
df["hour"]=df.times.dt.hour; df["minute"]=df.times.dt.minute
df["slot"]=df.hour*4+df.minute//15
df["date"]=df.times.dt.normalize()
df["month"]=df.times.dt.month; df["dow"]=df.times.dt.dayofweek
print("merged:", df.shape, "  price NA:", df["A"].isna().sum())

# net load (with 联络线)
df["net_load_act"] = df["系统负荷实际值"] - df["风光总加实际值"] - df["水电实际值"] - df["非市场化机组实际值"] - df["联络线实际值"]
df["net_load_fct"] = df["系统负荷预测值"] - df["风光总加预测值"] - df["水电预测值"] - df["非市场化机组预测值"] - df["联络线预测值"]""")

md("""## 数据质量与对齐

价格行少 117 行 → 看缺哪些时间。""")

co(r"""miss = df[df["A"].isna()]
print("price NA rows:", len(miss))
miss_per_day = miss.groupby("date").size().sort_values(ascending=False)
print("days with NA price (top):"); print(miss_per_day.head(10))
print("\nfull days (96 NA):", (miss_per_day==96).sum())
print("partial NA days   :", ((miss_per_day>0)&(miss_per_day<96)).sum())""")

md("""---
# A. 预测值质量（forecast vs actual）

每个 channel 看 MAE / RMSE / bias / 相对误差，以及随小时/月份的变化。""")

co(r"""rows=[]
for c in CH:
    a=df[c+"实际值"]; f=df[c+"预测值"]
    err = f-a
    rng = a.max()-a.min()
    rows.append(dict(channel=c,
        mean_actual=a.mean(), std_actual=a.std(),
        bias=err.mean(), mae=err.abs().mean(),
        rmse=float(np.sqrt((err**2).mean())),
        rel_mae_pct=err.abs().mean()/max(abs(a).mean(),1e-9)*100,
        nrmse_range_pct=float(np.sqrt((err**2).mean()))/max(rng,1e-9)*100,
    ))
chan_err = pd.DataFrame(rows).round(4)
chan_err.to_csv(OUT/"A_channel_error.csv", index=False)
chan_err""")

co(r"""# Bias / MAE 的小时分布（heatmap）
fig, axes = plt.subplots(2, 1, figsize=(11, 7))
H = pd.DataFrame({c: (df[c+"预测值"]-df[c+"实际值"]).groupby(df.hour).mean() for c in CH})
sns.heatmap(H.T, ax=axes[0], cmap="coolwarm", center=0, annot=True, fmt=".2f", cbar=False)
axes[0].set_title("Bias by hour (forecast − actual)"); axes[0].set_xlabel("")

H2 = pd.DataFrame({c: (df[c+"预测值"]-df[c+"实际值"]).abs().groupby(df.hour).mean() for c in CH})
sns.heatmap(H2.T, ax=axes[1], cmap="Reds", annot=True, fmt=".2f", cbar=False)
axes[1].set_title("MAE by hour"); axes[1].set_xlabel("hour")
plt.tight_layout(); plt.savefig(OUT/"A_error_by_hour.png"); plt.show()""")

co(r"""# Bias / MAE 的月份分布
fig, axes = plt.subplots(2, 1, figsize=(11, 7))
M  = pd.DataFrame({c: (df[c+"预测值"]-df[c+"实际值"]).groupby(df.month).mean() for c in CH})
sns.heatmap(M.T, ax=axes[0], cmap="coolwarm", center=0, annot=True, fmt=".2f", cbar=False)
axes[0].set_title("Bias by month"); axes[0].set_xlabel("")

M2 = pd.DataFrame({c: (df[c+"预测值"]-df[c+"实际值"]).abs().groupby(df.month).mean() for c in CH})
sns.heatmap(M2.T, ax=axes[1], cmap="Reds", annot=True, fmt=".2f", cbar=False)
axes[1].set_title("MAE by month"); axes[1].set_xlabel("month")
plt.tight_layout(); plt.savefig(OUT/"A_error_by_month.png"); plt.show()""")

co(r"""# 误差与实际值水平的关系：分位数分箱
import matplotlib.gridspec as gs
fig=plt.figure(figsize=(12,8))
G=gs.GridSpec(3,3,figure=fig)
for i,c in enumerate(CH):
    ax=fig.add_subplot(G[i//3,i%3])
    a=df[c+"实际值"]; e=df[c+"预测值"]-a
    bins=pd.qcut(a, 10, duplicates="drop")
    g=pd.DataFrame({"bin":bins,"err":e}).groupby("bin")["err"]
    m=g.mean(); s=g.std()
    x=np.arange(len(m))
    ax.bar(x, m, yerr=s, color="steelblue", alpha=.7)
    ax.axhline(0,color="k",lw=.5); ax.set_title(c, fontsize=10)
    ax.set_xticks([]); ax.set_xlabel("low → high actual"); ax.set_ylabel("bias")
plt.tight_layout(); plt.savefig(OUT/"A_bias_by_actual_level.png"); plt.show()""")

co(r"""# 净负荷的预测残差（这是最关键的派生量）
e = df["net_load_fct"] - df["net_load_act"]
print("net_load forecast: bias=%.3f mae=%.3f rmse=%.3f"%(e.mean(),e.abs().mean(),float(np.sqrt((e**2).mean()))))
fig,axes=plt.subplots(1,2,figsize=(11,3.2))
axes[0].hist(e, bins=80, color="steelblue"); axes[0].axvline(0,c="k",lw=1)
axes[0].set_title("net_load forecast residual"); axes[0].set_xlabel("forecast − actual")
hour_b=e.groupby(df.hour).mean(); hour_m=e.abs().groupby(df.hour).mean()
axes[1].plot(hour_b,label="bias",marker="o"); axes[1].plot(hour_m,label="MAE",marker="o")
axes[1].axhline(0,c="k",lw=.5); axes[1].set_xlabel("hour"); axes[1].legend()
axes[1].set_title("net_load residual by hour")
plt.tight_layout(); plt.savefig(OUT/"A_net_load_residual.png"); plt.show()""")

md("""---
# B. 实际电价结构""")

co(r"""price = df.dropna(subset=["A"]).copy()
print("rows w/ price:", len(price), "  full days:", (price.groupby("date").size()==96).sum())
print(price["A"].describe())
fig,axes=plt.subplots(1,2,figsize=(11,3.2))
axes[0].hist(price["A"], bins=80, color="darkorange"); axes[0].set_title("price distribution"); axes[0].set_xlabel("A")
price.groupby("date")["A"].mean().plot(ax=axes[1])
axes[1].set_title("daily mean price"); axes[1].set_xlabel("")
plt.tight_layout(); plt.savefig(OUT/"B_price_overview.png"); plt.show()""")

co(r"""# 日内形状（按月分组的均值曲线）
slots=np.arange(96)
fig,ax=plt.subplots(figsize=(11,4))
for m in range(1,13):
    sub=price[price.month==m]
    if sub.empty: continue
    curve=sub.groupby("slot")["A"].mean().reindex(slots)
    ax.plot(slots, curve, label=f"M{m}", alpha=.85)
ax.set_xlabel("15-min slot (0..95)"); ax.set_ylabel("mean price")
ax.set_title("Monthly mean intraday curve"); ax.legend(ncol=6, fontsize=8)
plt.tight_layout(); plt.savefig(OUT/"B_monthly_curve.png"); plt.show()""")

co(r"""# 日型聚类
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
mat = price.pivot_table(index="date", columns="slot", values="A")
mat = mat.dropna()  # full days only
print("full days for clustering:", len(mat))
# z-score per day → focus on shape
mat_z = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1).replace(0,1), axis=0)
K=5
km=KMeans(n_clusters=K, random_state=2026, n_init=10).fit(mat_z.values)
labels=pd.Series(km.labels_, index=mat.index, name="cluster")

fig,axes=plt.subplots(1,2,figsize=(13,4))
for k in range(K):
    sub_idx=labels[labels==k].index
    axes[0].plot(np.arange(96), mat_z.loc[sub_idx].mean(axis=0), label=f"C{k} (n={len(sub_idx)})", lw=2)
axes[0].set_title("Daily price shape clusters (z per day)"); axes[0].legend()
axes[0].set_xlabel("slot")

# 各聚类的月份分布
month_dist = pd.crosstab(labels.index.month, labels)
month_dist_pct = month_dist.div(month_dist.sum(axis=1), axis=0)
sns.heatmap(month_dist_pct, ax=axes[1], cmap="Blues", annot=True, fmt=".0%", cbar=False)
axes[1].set_title("Cluster share by month"); axes[1].set_xlabel("cluster"); axes[1].set_ylabel("month")
plt.tight_layout(); plt.savefig(OUT/"B_clusters.png"); plt.show()
labels.to_csv(OUT/"B_day_clusters.csv")""")

co(r"""# 峰谷价差与 oracle 收益分布
def oracle_day(p):
    p=np.asarray(p,float); BLOCK=8; N=96
    best=-np.inf; tc=td=0
    for c in range(0,81):
        for d in range(c+BLOCK, 89):
            spread = p[d:d+BLOCK].sum()-p[c:c+BLOCK].sum()
            if spread>best: best,tc,td=spread,c,d
    return best, tc, td
rows=[]
for date, day in price.groupby("date"):
    if len(day)!=96: continue
    p=day.sort_values("slot")["A"].to_numpy()
    sp,tc,td=oracle_day(p)
    rows.append(dict(date=date, oracle_spread=sp, tc=tc, td=td,
                      mean=p.mean(), std=p.std(), pmax=p.max(), pmin=p.min()))
oracle_df=pd.DataFrame(rows).set_index("date")
oracle_df.to_csv(OUT/"B_oracle_per_day.csv")
print(oracle_df["oracle_spread"].describe().round(2))

fig,axes=plt.subplots(1,3,figsize=(14,3.5))
axes[0].hist(oracle_df["oracle_spread"], bins=60, color="seagreen")
axes[0].set_title("Oracle daily spread (sum-block)")
axes[0].axvline(oracle_df["oracle_spread"].median(), c="k", ls="--")
oracle_df["oracle_spread"].rolling(7).mean().plot(ax=axes[1])
axes[1].set_title("7d rolling oracle spread"); axes[1].set_xlabel("")
axes[2].scatter(oracle_df["tc"], oracle_df["td"], s=5, alpha=.4)
axes[2].set_title("Oracle (tc, td) scatter"); axes[2].set_xlabel("charge_start"); axes[2].set_ylabel("discharge_start")
plt.tight_layout(); plt.savefig(OUT/"B_oracle.png"); plt.show()""")

md("""---
# C. 输入 → 价格关系""")

co(r"""# 净负荷 vs 价格
sub=price.dropna(subset=["A","net_load_act"]).sample(min(5000,len(price)), random_state=0)
fig,axes=plt.subplots(1,2,figsize=(12,3.8))
axes[0].scatter(sub["net_load_act"], sub["A"], s=4, alpha=.3)
b=pd.qcut(price["net_load_act"], 20, duplicates="drop")
binstat=price.groupby(b)["A"].agg(["mean","std","count"])
xs=binstat.index.map(lambda iv: iv.mid).to_numpy(dtype=float)
axes[0].errorbar(xs, binstat["mean"], yerr=binstat["std"]/np.sqrt(binstat["count"]),
                 fmt="o-", color="red", label="binned mean")
axes[0].set_xlabel("net_load (actual)"); axes[0].set_ylabel("price A"); axes[0].set_title("Price vs net load (with 联络线)"); axes[0].legend()

# 风光占比 vs 价格
ren_share = price["风光总加实际值"]/price["系统负荷实际值"].replace(0,np.nan)
b2=pd.qcut(ren_share, 20, duplicates="drop")
bs2=price.groupby(b2)["A"].agg(["mean","std","count"])
xs2=bs2.index.map(lambda iv: iv.mid).to_numpy(dtype=float)
axes[1].errorbar(xs2, bs2["mean"], yerr=bs2["std"]/np.sqrt(bs2["count"]), fmt="o-", color="darkgreen")
axes[1].set_xlabel("renewable share"); axes[1].set_ylabel("price A"); axes[1].set_title("Price vs renewable share")
plt.tight_layout(); plt.savefig(OUT/"C_price_vs_inputs.png"); plt.show()""")

co(r"""# 各 channel 实际值与价格的相关
import scipy.stats as st
rows=[]
for c in CH:
    a=price[c+"实际值"]; f=price[c+"预测值"]
    rows.append(dict(channel=c,
        corr_actual=st.pearsonr(a, price["A"])[0],
        corr_forecast=st.pearsonr(f, price["A"])[0],
        spearman_actual=st.spearmanr(a, price["A"])[0],
    ))
rows.append(dict(channel="net_load",
    corr_actual=st.pearsonr(price["net_load_act"], price["A"])[0],
    corr_forecast=st.pearsonr(price["net_load_fct"], price["A"])[0],
    spearman_actual=st.spearmanr(price["net_load_act"], price["A"])[0],
))
corr_df=pd.DataFrame(rows).round(3)
corr_df.to_csv(OUT/"C_corr_with_price.csv", index=False)
corr_df""")

co(r"""# 预测残差 vs 价格残差（去日均）
g_mean = price.groupby("date")["A"].transform("mean")
price_dev = price["A"] - g_mean
print("predicting daily-deviation of price using residuals (forecast errors)…")
err = pd.DataFrame({c: (price[c+"预测值"]-price[c+"实际值"]) for c in CH})
err["net"] = price["net_load_fct"]-price["net_load_act"]
corr_resid = err.corrwith(price_dev).round(3).sort_values()
print(corr_resid)
corr_resid.to_csv(OUT/"C_residual_vs_price_dev.csv")""")

md("""---
# D. 训练-测试分布偏移 + 春节

`test` 只有预测值，所以这里**只能比预测值的分布**（与 train 中的预测值列对齐）。""")

co(r"""# 训练全年 vs 训练最后30天 vs 测试，按预测值分布看
def summary(d, label):
    return d[FCT].agg(["mean","std","min","max"]).T.assign(label=label)

train_all = bnd
train_tail= bnd[bnd.times >= bnd.times.max() - pd.Timedelta(days=30)]
test_only = test.copy()

print("train all  :", len(train_all), train_all.times.min(),"~",train_all.times.max())
print("train tail :", len(train_tail), train_tail.times.min(),"~",train_tail.times.max())
print("test       :", len(test_only), test_only.times.min(),"~", test_only.times.max())

stat = pd.concat([summary(train_all,"train_all"),
                  summary(train_tail,"train_tail30d"),
                  summary(test_only,"test")]).reset_index().rename(columns={"index":"channel"})
stat.to_csv(OUT/"D_distribution_summary.csv", index=False)
stat""")

co(r"""# KS 距离: train_tail vs test
from scipy.stats import ks_2samp
rows=[]
for c in FCT:
    a=train_tail[c].dropna().to_numpy(); b=test_only[c].dropna().to_numpy()
    ks=ks_2samp(a,b)
    rows.append(dict(channel=c, ks_stat=ks.statistic, ks_p=ks.pvalue,
                     mean_tail=a.mean(), mean_test=b.mean(),
                     pct_test_above_train_max= float((b>a.max()).mean()),
                     pct_test_below_train_min= float((b<a.min()).mean())))
ks_df=pd.DataFrame(rows).round(4)
ks_df.to_csv(OUT/"D_ks_tail_vs_test.csv", index=False)
ks_df""")

co(r"""# 月度风/光峰值（量化产能扩张）
peaks = bnd.assign(month=bnd.times.dt.month).groupby("month")[ACT].max()
peaks.to_csv(OUT/"D_monthly_peak_actual.csv")
fig,ax=plt.subplots(figsize=(11,3.5))
peaks[["风电实际值","光伏实际值","风光总加实际值"]].plot(marker="o", ax=ax)
ax.set_title("Monthly max actual output (capacity proxy)"); ax.set_ylabel("max p.u.")
plt.tight_layout(); plt.savefig(OUT/"D_monthly_peak.png"); plt.show()""")

co(r"""# 春节场景: 2025-01-28..02-04
sf_start=pd.Timestamp("2025-01-28"); sf_end=pd.Timestamp("2025-02-04 23:59:59")
sf = price[(price.times>=sf_start)&(price.times<=sf_end)]
non_sf_jan_feb = price[(price.month.isin([1,2])) & ~((price.times>=sf_start)&(price.times<=sf_end))]
mar_apr = price[price.month.isin([3,4])]

def desc(d, lab):
    return dict(label=lab, days=d["date"].nunique(),
                price_mean=d["A"].mean(), price_std=d["A"].std(),
                ren_share_mean=(d["风光总加实际值"]/d["系统负荷实际值"]).mean(),
                load_mean=d["系统负荷实际值"].mean())
sf_summary = pd.DataFrame([desc(sf,"spring_festival_2025"),
                           desc(non_sf_jan_feb,"jan_feb_excl_sf"),
                           desc(mar_apr,"mar_apr_2025")]).round(3)
sf_summary.to_csv(OUT/"D_spring_festival.csv", index=False)
print(sf_summary)

# 春节 vs 非春节的日内曲线对比
fig,ax=plt.subplots(figsize=(11,3.5))
for d, lab in [(sf,"spring_festival"),(non_sf_jan_feb,"jan_feb_excl_sf"),(mar_apr,"mar_apr")]:
    if d.empty: continue
    curve=d.groupby("slot")["A"].mean()
    ax.plot(curve.index, curve.values, label=lab, lw=2)
ax.set_title("Intraday price: spring festival vs surroundings"); ax.legend(); ax.set_xlabel("slot")
plt.tight_layout(); plt.savefig(OUT/"D_spring_festival.png"); plt.show()""")

md("""---
# E. Oracle 调度难度

把每日 oracle profit 与 (日内方差 / 净负荷形状 / 风光占比) 对照。""")

co(r"""# Oracle 收益分解：哪些日子最赚 / 最难
day_feat = price.groupby("date").agg(
    price_mean=("A","mean"), price_std=("A","std"),
    load_mean=("系统负荷实际值","mean"),
    ren_share=("风光总加实际值",lambda s: (s/price.loc[s.index,"系统负荷实际值"]).mean()),
    nlf_mae=("net_load_fct", lambda s: (s-price.loc[s.index,"net_load_act"]).abs().mean()),
).join(oracle_df[["oracle_spread","tc","td"]], how="inner")
day_feat["month"]=day_feat.index.month
day_feat.to_csv(OUT/"E_day_features.csv")
print(day_feat.describe().round(2))

# 散点：oracle_spread vs price_std / ren_share / nlf_mae
fig,axes=plt.subplots(1,3,figsize=(14,3.6))
for ax,(x,lab) in zip(axes, [("price_std","intraday std"),("ren_share","ren share"),("nlf_mae","net_load forecast MAE")]):
    ax.scatter(day_feat[x], day_feat["oracle_spread"], s=10, alpha=.5)
    ax.set_xlabel(lab); ax.set_ylabel("oracle spread"); ax.set_title(f"oracle vs {lab}")
plt.tight_layout(); plt.savefig(OUT/"E_oracle_scatter.png"); plt.show()""")

co(r"""# Oracle (tc, td) 的高频组合 → 先验
combo = day_feat.groupby(["tc","td"]).size().sort_values(ascending=False).head(15).rename("count").reset_index()
combo.to_csv(OUT/"E_top_oracle_slots.csv", index=False)
print(combo)
fig,axes=plt.subplots(1,2,figsize=(11,3.5))
day_feat["tc"].value_counts().sort_index().plot.bar(ax=axes[0]); axes[0].set_title("oracle charge_start hist")
day_feat["td"].value_counts().sort_index().plot.bar(ax=axes[1]); axes[1].set_title("oracle discharge_start hist")
plt.tight_layout(); plt.savefig(OUT/"E_oracle_slots.png"); plt.show()""")

md("""---
# F. 结论与改进方向

将在 notebook 执行后，根据上面 A~E 的真实输出在此填写要点。""")

nb.cells = cells
out_path = Path("/Users/yw/ai/electricity/reports/eda.ipynb")
nbf.write(nb, out_path)
print(f"wrote {out_path} with {len(cells)} cells")
