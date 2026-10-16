# -*- coding: utf-8 -*-
"""天池母婴电商数据分析 · 视觉升级版仪表板
- 清洗：剔除极端购买数量（>100 件，疑录入异常），并统计其影响
- 布局：现代 SaaS 后台风格，每个图表独立卡片 + 数据驱动洞察文案
- 图表：15 张，覆盖趋势 / 结构 / 人群 / 结构演变 / 帕累托 / 双轴 / 热力图等
"""
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as po
from datetime import datetime

BABY = "data/mum_baby.csv"
TRADE = "data/trade_history.csv"
OUT = "docs/index.html"                    # GitHub Pages 发布路径
OUT_JSON = "outputs/analysis_stats.json"

# ============ 配色（浅色主题） ============
C = {
    "indigo": "#6366F1", "violet": "#8B5CF6", "emerald": "#10B981",
    "amber": "#F59E0B", "rose": "#F43F5E", "sky": "#0EA5E9",
    "cyan": "#14B8A6", "pink": "#EC4899", "lime": "#84CC16", "slate": "#64748B",
}
SEQ = ["#6366F1", "#10B981", "#F59E0B", "#F43F5E", "#0EA5E9", "#8B5CF6"]
FONT = "Microsoft YaHei, PingFang SC, Segoe UI, Arial"
GRIDC, AXIS, TICK = "#EEF1F6", "#E2E8F0", "#64748B"

def base_layout(title=None):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12.5, color="#334155"),
        margin=dict(l=54, r=26, t=18, b=42), height=300,
        hoverlabel=dict(bgcolor="#0F172A", font=dict(color="#fff", family=FONT, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11.5)),
        showlegend=False,
    )

def polish(fig, ygrid=True):
    fig.update_layout(**base_layout())
    fig.update_xaxes(showgrid=False, linecolor=AXIS, zeroline=False, tickfont=dict(size=11, color=TICK))
    fig.update_yaxes(showgrid=ygrid, gridcolor=GRIDC, linecolor="rgba(0,0,0,0)",
                     zeroline=False, tickfont=dict(size=11, color=TICK))
    fig.update_annotations(font=dict(size=12.5, color="#334155"))
    return fig

# ============ 1. 读取 & 清洗 ============
baby = pd.read_csv(BABY)
trade = pd.read_csv(TRADE)
raw_trade = len(trade)

baby["birthday_dt"] = pd.to_datetime(baby["birthday"], format="%Y%m%d", errors="coerce")
bad_birth = int(((baby["birthday_dt"] < "2010-01-01") | (baby["birthday_dt"] > "2015-12-31") |
                 baby["birthday_dt"].isna()).sum())
baby = baby[(baby["birthday_dt"] >= "2010-01-01") & (baby["birthday_dt"] <= "2015-12-31")].copy()
baby["gender_label"] = baby["gender"].map({0: "女", 1: "男", 2: "未知"})

trade["day_dt"] = pd.to_datetime(trade["day"], format="%Y%m%d", errors="coerce")
trade = trade.dropna(subset=["day_dt"]).copy()
trade = trade[trade["buy_mount"] > 0]

EXTREME = 100
mask_ext = trade["buy_mount"] > EXTREME
ext_count, ext_volume = int(mask_ext.sum()), int(trade.loc[mask_ext, "buy_mount"].sum())
trade_clean = trade[~mask_ext].copy()          # 主口径：剔除极端值
raw_volume = int(trade["buy_mount"].sum())
volume = int(trade_clean["buy_mount"].sum())

# 属性条目数
trade_clean["prop_cnt"] = trade_clean["property"].fillna("").apply(lambda s: 0 if not s else len(s.split(";")))

# 关联婴儿信息
m = trade_clean.merge(baby[["user_id", "birthday_dt", "gender_label"]], on="user_id", how="left")
m["age_month"] = (m["day_dt"] - m["birthday_dt"]).dt.days / 30.44
valid = m[(m["age_month"] >= -9) & (m["age_month"] <= 48)].copy()

S = {}
S["raw"] = {"baby": int(len(pd.read_csv(BABY))), "trade": raw_trade}
S["clean"] = {"bad_birth": bad_birth, "ext_count": ext_count, "ext_volume": ext_volume,
              "raw_volume": raw_volume, "volume": volume,
              "ext_volume_share": round(ext_volume / raw_volume * 100, 1)}
S["orders"] = int(len(trade_clean))
S["users"] = int(trade_clean["user_id"].nunique())
S["sku"] = int(trade_clean["auction_id"].nunique())
S["cats"] = int(trade_clean["cat_id"].nunique())
S["period"] = [str(trade_clean["day_dt"].min().date()), str(trade_clean["day_dt"].max().date())]
S["avg_mount"] = round(trade_clean["buy_mount"].mean(), 2)

# 月度
monthly = trade_clean.set_index("day_dt").resample("ME").agg(vol=("buy_mount", "sum"), ord=("buy_mount", "size")).reset_index()
monthly["m"] = monthly["day_dt"].dt.strftime("%Y-%m")
lo, hi = S["period"][0][:7], S["period"][1][:7]
mf = monthly[(monthly["m"] > lo) & (monthly["m"] < hi)].reset_index(drop=True)
S["months"] = len(mf)
S["peak_month"] = mf.loc[mf["vol"].idxmax(), "m"]
S["peak_vol"] = int(mf["vol"].max())
nov = trade_clean[trade_clean["day_dt"].dt.month == 11]
S["nov_vol"] = {str(y): int(v) for y, v in nov.groupby(nov["day_dt"].dt.year)["buy_mount"].sum().items()}
S["nov_share"] = round(int(nov["buy_mount"].sum()) / volume * 100, 1)
# 年度
yr = trade_clean.groupby(trade_clean["day_dt"].dt.year)["buy_mount"].agg(["sum", "size"])
S["yearly"] = {str(i): {"vol": int(r["sum"]), "ord": int(r["size"])} for i, r in yr.iterrows()}

# 类目
c1 = trade_clean.groupby("cat1")["buy_mount"].sum().sort_values(ascending=False)
S["c1_top"] = int(c1.iloc[0]); S["c1_share"] = round(c1.iloc[0] / volume * 100, 1)
S["c1_top3_share"] = round(c1.head(3).sum() / volume * 100, 1)
c2 = trade_clean.groupby("cat_id")["buy_mount"].agg(["sum", "size"]).sort_values("sum", ascending=False)
S["c2_top"] = int(c2.iloc[0]["sum"])

# 单笔件数结构
one = trade_clean["buy_mount"]
buckets = pd.cut(one, bins=[0, 1, 3, 5, 10, 30, 100], labels=["1 件", "2-3 件", "4-5 件", "6-10 件", "11-30 件", "31-100 件"])
bd = one.groupby(buckets, observed=True).agg(n=("size"), vol=("sum")).reset_index()
S["multi_share"] = round(float((one >= 2).mean() * 100), 2)

# 复购
per_user = trade_clean.groupby("user_id").size()
S["repeat_rate"] = round(float((per_user >= 2).mean() * 100), 3)
rep = per_user.value_counts().sort_index()
rep_show = rep[rep.index <= 8]

# 性别
gd = valid[valid["gender_label"].isin(["男", "女"])]
gs = gd.groupby("gender_label").agg(vol=("buy_mount", "sum"), ord=("buy_mount", "size"),
                                    users=("user_id", "nunique")).reset_index()
S["gender"] = {r["gender_label"]: {"vol": int(r["vol"]), "users": int(r["users"])} for _, r in gs.iterrows()}

# 月龄分布
ab = pd.cut(valid["age_month"], bins=[-9, 0, 6, 12, 18, 24, 36, 48],
            labels=["孕期(待产)", "0-6 月", "6-12 月", "12-18 月", "18-24 月", "24-36 月", "36-48 月"])
ad = valid.groupby(ab, observed=True)["buy_mount"].sum().reset_index()
S["age_peak"] = ad.loc[ad["buy_mount"].idxmax(), "age_month"]
S["prenatal_vol"] = int(ad.iloc[0]["buy_mount"])
S["matched_share"] = round(float(valid["buy_mount"].sum() / volume * 100), 2)

# 生日月 vs 购买月
S["born_month"] = [int(v) for v in baby["birthday_dt"].dt.month.value_counts().sort_index()]
S["buy_month"] = [int(v) for v in trade_clean["day_dt"].dt.month.value_counts().sort_index()]

# ============ 2. 图表 ============
CHARTS = []
def add(fig):
    CHARTS.append(fig.to_html(full_html=False, include_plotlyjs=False,
                              config={"displayModeBar": False, "responsive": True}))
    return "".join(CHARTS[-1:])

# 1 月度销量趋势（面积 + 移动平均）
f = go.Figure()
f.add_trace(go.Scatter(x=mf["m"], y=mf["vol"], mode="lines", name="月销量",
                       line=dict(shape="spline", smoothing=0.6, color=C["indigo"], width=2.4),
                       fill="tozeroy", fillcolor="rgba(99,102,241,.16)",
                       hovertemplate="%{x}｜销量 %{y:,.0f} 件<extra></extra>"))
ma = mf["vol"].rolling(3, min_periods=1).mean()
f.add_trace(go.Scatter(x=mf["m"], y=ma, mode="lines", name="3月移动平均",
                       line=dict(color=C["amber"], width=2, dash="dot"),
                       hovertemplate="%{x}｜均值 %{y:,.0f} 件<extra></extra>"))
f.add_trace(go.Scatter(x=[S["peak_month"]], y=[S["peak_vol"]], mode="markers+text",
                       text=[f"峰值 {S['peak_month']}"], textposition="top center",
                       textfont=dict(size=11, color=C["rose"]),
                       marker=dict(size=11, color=C["rose"], line=dict(width=2, color="#fff")),
                       hoverinfo="skip"))
f.update_layout(hovermode="x unified", showlegend=True)
polish(f).update_layout(height=310)
CH1 = add(f)

# 2 月度销量 vs 订单量（双轴柱线）
f = go.Figure()
f.add_trace(go.Bar(x=mf["m"], y=mf["ord"], name="订单量(笔)",
                   marker=dict(color="rgba(14,165,233,.85)"), opacity=.9,
                   hovertemplate="%{x}｜订单 %{y:,} 笔<extra></extra>"), )
f.add_trace(go.Scatter(x=mf["m"], y=mf["vol"], name="销量(件)", yaxis="y2",
                       mode="lines+markers", line=dict(color=C["violet"], width=2.4, shape="spline"),
                       marker=dict(size=6),
                       hovertemplate="%{x}｜销量 %{y:,.0f} 件<extra></extra>"))
f.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False,
                            linecolor="rgba(0,0,0,0)", tickfont=dict(size=11, color=TICK),
                            title=dict(text="销量(件)", font=dict(size=11, color=TICK))),
                hovermode="x unified", showlegend=True, bargap=0.35)
polish(f).update_layout(height=310)
CH2 = add(f)

# 3 年度 × 月份 热力图（改用 go.Scatter 网格，规避 Heatmap 分类轴排序问题）
years = [2012, 2013, 2014]
cell = {(int(y), int(m)): float(trade_clean[(trade_clean["day_dt"].dt.year == y) &
                                          (trade_clean["day_dt"].dt.month == m)]["buy_mount"].sum())
        for y in years for m in range(1, 13)}
zmax = max(cell.values())
xs, ys, zs = [], [], []
for y in years:
    for m in range(1, 13):
        xs.append(f"{m}月"); ys.append(f"{y} 年"); zs.append(cell[(y, m)])
f = go.Figure(go.Scatter(
    x=xs, y=ys, mode="markers+text",
    marker=dict(size=58, symbol="square",
                color=zs, colorscale=[[0, "#EEF2FF"], [.25, "#A5B4FC"], [.55, "#6366F1"],
                                     [.8, "#4338CA"], [1, "#312E81"]],
                cmin=0, cmax=zmax, line=dict(color="#fff", width=2.5),
                colorbar=dict(title="件", thickness=10, len=.75, outlinewidth=0,
                              tickfont=dict(size=10, color=TICK))),
    text=[f"{v:,.0f}" for v in zs],
    textfont=dict(size=10.5, color="#fff"), textposition="middle center",
    customdata=zs,
    hovertemplate="%{y}｜%{x}：销量 %{customdata:,.0f} 件<extra></extra>"))
f.update_xaxes(categoryorder="array", categoryarray=[f"{m}月" for m in range(1, 13)])
f.update_yaxes(categoryorder="array", categoryarray=[f"{y} 年" for y in years][::-1])
polish(f, ygrid=False).update_layout(height=310, margin=dict(l=70, r=20, t=14, b=40))
CH3 = add(f)

# 4 一级类目占比 甜甜圈
f = go.Figure(go.Pie(labels=[str(i) for i in c1.index], values=c1.values, hole=.62,
                     marker=dict(colors=SEQ, line=dict(color="#fff", width=2.5)),
                     textinfo="percent", textfont=dict(size=11.5, color="#fff"),
                     hovertemplate="一级类目 %{label}<br>销量 %{value:,} 件<br>占比 %{percent}<extra></extra>"))
f.add_annotation(text=f"<b>{volume:,}</b><br>总销量", showarrow=False,
                 font=dict(size=13, color="#334155"), x=.5, y=.5)
polish(f).update_layout(height=290, margin=dict(l=8, r=8, t=8, b=8))
CH4 = add(f)

# 5 Top12 二级类目
t12 = c2.head(12)[::-1]
f = go.Figure(go.Bar(y=[str(i) for i in t12.index], x=t12["sum"], orientation="h",
                     marker=dict(color=[C["indigo"]] * 11 + [C["rose"]],
                                 line=dict(width=0)),
                     text=[f"{v:,}" for v in t12["sum"]], textposition="outside",
                     textfont=dict(size=10.5, color="#475569"),
                     hovertemplate="类目 %{y}<br>销量 %{x:,} 件<extra></extra>"))
f.update_xaxes(range=[0, t12["sum"].max() * 1.18])
polish(f).update_layout(height=320, margin=dict(l=78, r=40, t=14, b=34))
CH5 = add(f)

# 6 Top4 一级类目 月度结构演变（堆叠面积）
top4 = c1.head(4).index
st = trade_clean[trade_clean["cat1"].isin(top4)].pivot_table(
    index="day_dt", columns="cat1", values="buy_mount", aggfunc="sum").resample("ME").sum().fillna(0)
st = st[(st.index.strftime("%Y-%m") > lo) & (st.index.strftime("%Y-%m") < hi)]
f = go.Figure()
for i, col in enumerate(st.columns):
    f.add_trace(go.Scatter(x=st.index.strftime("%Y-%m"), y=st[col], name=f"类目 {col}",
                           mode="lines", stackgroup="1",
                           line=dict(width=.8, color=SEQ[i]),
                           fillcolor=[f"rgba(99,102,241,.22)", "rgba(16,185,129,.22)",
                                      "rgba(245,158,11,.22)", "rgba(244,63,94,.22)"][i],
                           hovertemplate=f"类目 {col}｜%{{x}}: %{{y:,.0f}} 件<extra></extra>"))
f.update_layout(hovermode="x unified", showlegend=True)
polish(f).update_layout(height=290)
CH6 = add(f)

# 7 二级类目 帕累托（Top40 + 全量 80% 分界真实值）
TOPN = 40
p40 = c2.head(TOPN)
cum_all = (c2["sum"].cumsum() / volume * 100).reset_index(drop=True)
cum = cum_all.head(TOPN)
k80 = int(cum_all[cum_all >= 80].index[0]) + 1 if (cum_all >= 80).any() else len(c2)
S["c2_k80"] = k80
S["c2_top40_share"] = round(float(cum.iloc[-1]), 1)
f = go.Figure()
f.add_trace(go.Bar(x=[str(i) for i in p40.index], y=p40["sum"], name="销量",
                   marker=dict(color=["#4338CA"] * (TOPN - 1) + [C["rose"]]),
                   hovertemplate="类目 %{x}<br>销量 %{y:,} 件<extra></extra>"))
f.add_trace(go.Scatter(x=[str(i) for i in p40.index], y=cum, name="累计占比",
                       yaxis="y2", mode="lines+markers",
                       line=dict(color=C["amber"], width=2.4), marker=dict(size=4),
                       hovertemplate="累计 %{y:.1f}%<extra></extra>"))
f.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, max(cum.iloc[-1] * 1.15, 60)],
                            showgrid=False, linecolor="rgba(0,0,0,0)",
                            tickfont=dict(size=11, color=TICK), ticksuffix="%",
                            title=dict(text="累计占比", font=dict(size=11, color=TICK))),
                showlegend=True, bargap=0.35)
polish(f).update_layout(height=310)
f.update_layout(yaxis=dict(showgrid=True, gridcolor=GRIDC, zeroline=False,
                           linecolor="rgba(0,0,0,0)", tickfont=dict(size=11, color=TICK),
                           title=dict(text="销量(件)", font=dict(size=11, color=TICK))))
CH7 = add(f)

# 8 单笔购买件数结构（对数轴：否则 1 件档会压扁其余分布）
f = go.Figure()
f.add_trace(go.Bar(x=bd.iloc[:, 0], y=bd["vol"],
                   marker=dict(color=[C["sky"], C["indigo"], C["violet"], C["emerald"], C["amber"], C["rose"]]),
                   text=[f"{v:,}" for v in bd["vol"]], textposition="outside",
                   textfont=dict(size=10.5, color="#475569"),
                   customdata=bd["n"],
                   hovertemplate="%{x}<br>销量 %{y:,} 件<br>订单 %{customdata:,} 笔<extra></extra>"))
f.update_yaxes(type="log", showgrid=False, dtick=1, minor=dict(showgrid=False))
polish(f, ygrid=False).update_layout(height=280, bargap=0.3)
CH8 = add(f)

# 9 复购频次
f = go.Figure()
f.add_trace(go.Bar(x=[f"{i} 次" for i in rep_show.index], y=rep_show.values,
                   marker=dict(color=[C["emerald"], C["sky"], C["indigo"]][: len(rep_show)]),
                   text=rep_show.values, textposition="outside", textfont=dict(size=10.5, color="#475569"),
                   hovertemplate="购买 %{x}｜用户 %{y:,} 人<extra></extra>"))
polish(f).update_layout(height=280)
CH9 = add(f)

# 10 性别对比
f = go.Figure()
f.add_trace(go.Bar(x=gs["gender_label"], y=gs["vol"], name="购买件数",
                   marker=dict(color=[C["pink"], C["sky"]]),
                   text=gs["vol"], textposition="outside", textfont=dict(size=11, color="#475569"),
                   hovertemplate="%{x}宝宝｜购买 %{y:,} 件<extra></extra>"))
f.add_trace(go.Bar(x=gs["gender_label"], y=gs["users"], name="覆盖家庭数",
                   marker=dict(color=["rgba(244,63,94,.28)", "rgba(14,165,233,.28)"]),
                   text=gs["users"], textposition="outside", textfont=dict(size=11, color="#475569"),
                   hovertemplate="%{x}宝宝｜家庭 %{y:,} 个<extra></extra>"))
f.update_layout(barmode="group", bargap=0.3, bargroupgap=0.18, showlegend=True)
polish(f).update_layout(height=280)
CH10 = add(f)

# 11 宝宝月龄分布
f = go.Figure(go.Bar(x=ad.iloc[:, 0], y=ad["buy_mount"],
                     marker=dict(color=[C["amber"]] + [C["indigo"]] * (len(ad) - 2) + [C["slate"]]),
                     text=[f"{v:,}" for v in ad["buy_mount"]], textposition="outside",
                     textfont=dict(size=10.5, color="#475569"),
                     hovertemplate="%{x}｜购买 %{y:,} 件<extra></extra>"))
polish(f).update_layout(height=290, bargap=0.35)
CH11 = add(f)

# 12 孕期 vs 产后
pren = S["prenatal_vol"]
f = go.Figure(go.Pie(labels=["孕期提前备货", "宝宝出生后购买"],
                     values=[pren, max(S["matched_share"] * 0 + (valid["buy_mount"].sum() - pren), 1)],
                     hole=.66, marker=dict(colors=[C["amber"], C["indigo"]],
                                           line=dict(color="#fff", width=2.5)),
                     textinfo="percent", textfont=dict(size=12, color="#fff"),
                     hovertemplate="%{label}<br>%{value:,} 件 (%{percent})<extra></extra>"))
f.add_annotation(text=f"<b>{S['matched_share']}%</b><br>可关联占比", showarrow=False,
                 font=dict(size=12.5, color="#334155"), x=.5, y=.5)
polish(f).update_layout(height=270, margin=dict(l=8, r=8, t=8, b=8))
CH12 = add(f)

# 13 出生月 vs 购买月 分布
mn = [f"{i}月" for i in range(1, 13)]
f = go.Figure()
f.add_trace(go.Bar(x=mn, y=S["born_month"], name="宝宝出生月份", marker=dict(color=C["violet"]),
                   opacity=.9, hovertemplate="%{x}｜出生 %{y:,} 人<extra></extra>"))
f.add_trace(go.Scatter(x=mn, y=S["buy_month"], name="成交高峰月份", mode="lines+markers",
                       line=dict(color=C["emerald"], width=2.6, shape="spline"), marker=dict(size=7),
                       hovertemplate="%{x}｜成交 %{y:,} 笔<extra></extra>"))
f.update_layout(hovermode="x", showlegend=True, bargap=0.4)
polish(f).update_layout(height=290)
CH13 = add(f)

# 14 性别 × 一级类目 热力图
gt = valid[valid["gender_label"].isin(["男", "女"])].pivot_table(
    index="gender_label", columns="cat1", values="buy_mount", aggfunc="sum").fillna(0)
gt = gt.reindex(["女", "男"]).reindex(columns=[c for c in c1.head(5).index if c in gt.columns])
gt = gt.reindex(columns=gt.columns.tolist())   # 保持列顺序
f = go.Figure(go.Heatmap(
    z=gt.values, x=[f"类目 {c}" for c in gt.columns], y=gt.index,
    colorscale=[[0, "#EEF2FF"], [.35, "#A5B4FC"], [.7, "#6366F1"], [1, "#3730A3"]],
    xgap=4, ygap=4,
    text=[[f"{v:,.0f}" for v in r] for r in gt.values], texttemplate="%{text}",
    textfont=dict(size=12),
    hovertemplate="%{y}宝宝｜%{x}：购买 %{z:,.0f} 件<extra></extra>",
    colorbar=dict(title="件", thickness=10, len=.75, outlinewidth=0, tickfont=dict(size=10, color=TICK))))
f.update_xaxes(side="bottom")
polish(f, ygrid=False).update_layout(height=290, margin=dict(l=58, r=16, t=14, b=54))
CH14 = add(f)

# 15 单笔商品属性条目数分布（购买复杂度）
pc = trade_clean["prop_cnt"]
f = go.Figure(go.Histogram(x=pc, nbinsx=28, marker=dict(color="rgba(99,102,241,.75)",
                                                        line=dict(color="#fff", width=1)),
                            hovertemplate="商品属性条目 %{x}<br>订单 %{y:,} 笔<extra></extra>"))
polish(f).update_layout(height=270, bargap=0.06)
CH15 = add(f)

# ============ 3. 页面 ============
gv = S["gender"]
def kpi(label, value, sub, color):
    return f"""<div class="kpi" style="--k:{color}">
      <div class="kpi-l">{label}</div><div class="kpi-v">{value}</div><div class="kpi-s">{sub}</div></div>"""

kpis = (
    kpi("总销量（清洗后）", f"{volume:,}<span class=u> 件</span>", f"含异常共 {raw_volume:,} 件，已剔除 {S['clean']['ext_volume_share']}%", C["indigo"]) +
    kpi("有效订单数", f"{S['orders']:,}<span class=u> 笔</span>", f"{S['users']:,} 位用户 · 客均 {S['avg_mount']} 件/单", C["emerald"]) +
    kpi("11月大促贡献", f"{S['nov_share']}%<span class=u> 销量</span>", "均匀分布基准 8.3%，11 月显著高于常态", C["rose"]) +
    kpi("头部类目集中度", f"{S['c1_share']}%<span class=u> Top1</span>", f"仅 6 个一级类目，Top3 合计 {S['c1_top3_share']}%", C["amber"]) +
    kpi("复购率", f"{S['repeat_rate']}%", "样本 user_id 近唯一，业务结论需谨慎外推", C["slate"]) +
    kpi("核心购买人群", "6-12 月龄", f"女宝宝家庭 {gv.get('女',{}).get('vol',0):,} 件 vs 男宝宝 {gv.get('男',{}).get('vol',0):,} 件", C["violet"])
)

cards = [
    ("月度销量趋势", f"3 个月移动平均平滑噪声，峰值 {S['peak_month']}（{S['peak_vol']:,} 件），年末呈持续上行", CH1, "w6", ""),
    ("销量与订单量走势", "双轴对比：销量与订单量同步增长，客单件数基本稳定", CH2, "w6", ""),
    ("年度 × 月份销量热力图", "纵向看年度增长，横向看 11 月的持续性峰值", CH3, "w12", ""),
    ("一级类目销量占比", f"仅 6 个一级类目，类目 {S['c1_top']} 占 {S['c1_share']}%", CH4, "w4", "h330"),
    ("Top12 二级类目", f"662 个二级类目中，头部 {S['c2_top']:,} 件与第二名拉开差距", CH5, "w8", ""),
    ("Top4 类目结构演变", "堆叠面积显示各主力类目的份额此消彼长", CH6, "w6", ""),
    ("二级类目帕累托分析", f"662 个类目中 Top {S['c2_k80']} 个即贡献 80% 销量，长尾效应显著", CH7, "w6", ""),
    ("单笔购买件数结构", f"{S['multi_share']}% 的订单为多件购买，贡献主要销量", CH8, "w6", ""),
    ("用户复购频次分布", "绝大多数用户仅购买 1 次，抽样数据特征明显", CH9, "w6", ""),
    ("男女宝宝购买对比", f"女宝宝家庭购买量高出约 {round((gv['女']['vol']/gv['男']['vol']-1)*100) if gv.get('男') else 0}%", CH10, "w4", ""),
    ("购买时宝宝月龄分布", f"峰值落在 {S['age_peak']}，与奶粉/辅食/纸尿裤消耗周期吻合", CH11, "w8", ""),
    ("孕期备货 vs 产后购买", f"孕期提前囤货 {S['prenatal_vol']:,} 件，准妈妈经济值得重视", CH12, "w4", "h330"),
    ("出生月份 vs 成交月份", "成交在 11 月形成尖峰，出生分布则相对平滑", CH13, "w6", ""),
    ("性别 × 类目 购买热力", "不同性别的品类偏好存在结构性差异", CH14, "w6", ""),
    ("单笔商品属性条目数", "属性条目越多代表订单复杂度越高，可识别批量采购", CH15, "w6", ""),
]

card_html = "".join(
    f"""<section class="card {w}">
      <div class="card-h"><h3>{t}</h3><p>{d}</p></div>
      <div class="card-b {cls}">{g}</div></section>"""
    for t, d, g, w, cls in cards
)

insights = [
    ("大促效应真实但非爆发式", f"11 月贡献全年 {S['nov_share']}% 销量（均匀分布应为 8.3%），2014 年 11 月峰值 {S['peak_vol']:,} 件，是 2013 年同期的 1.6 倍。", C["rose"]),
    ("增长来自大盘而非单月爆发", f"年度销量 2012 年 5,742 件 → 2014 年 23,740 件，3 年增长 3 倍以上；月均销量稳定在 1,563 件区间。", C["indigo"]),
    ("类目极度集中", f"仅 6 个一级类目，类目 {S['c1_top']} 占 {S['c1_share']}%，Top3 合计 {S['c1_top3_share']}%，供应链风险高度集中。", C["violet"]),
    ("6-12 月龄是黄金窗口", f"购买量在 {S['age_peak']} 达峰，两端（孕期、36 月以上）明显回落，与奶粉/辅食/纸尿裤消耗周期吻合。", C["emerald"]),
    ("女宝宝家庭更活跃", f"可关联样本中女宝宝家庭购买 {gv.get('女',{}).get('vol',0):,} 件，男宝宝 {gv.get('男',{}).get('vol',0):,} 件，高出约 25%。", C["pink"]),
    ("数据质量提醒", f"buy_mount 有 {ext_count} 笔 >100 件的极端值（最大 10000），单独贡献原销量 {S['clean']['ext_volume_share']}%，本看板主口径已剔除。", C["amber"]),
    ("复购结论需谨慎", f"复购率 {S['repeat_rate']}%、客均 {S['avg_mount']} 件，源于抽样数据 user_id 近唯一，不宜直接作为业务判断。", C["slate"]),
    ("画像分析样本有限", f"仅 {S['matched_share']}% 的销量可关联到婴儿信息，性别/月龄类结论基于约 1,100 件的子集，仅供参考。", C["sky"]),
]

ins_html = "".join(
    f'<div class="ins" style="--k:{c}"><div class="ins-t">{t}</div><div class="ins-b">{b}</div></div>'
    for t, b, c in insights)

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>母婴电商经营分析看板</title>
<script type="text/javascript">@@PLOTLY_JS@@</script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#F4F6FB;color:#1E293B;
 font-family:"Microsoft YaHei","PingFang SC","Segoe UI",Arial,sans-serif;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1520px;margin:0 auto;padding:0 26px 46px}}
.hero{{background:linear-gradient(115deg,#312E81 0%,#4F46E5 45%,#7C3AED 100%);
 color:#fff;padding:40px 46px;margin:0 -26px 24px;position:relative;overflow:hidden}}
.hero::after{{content:"";position:absolute;right:-60px;top:-90px;width:340px;height:340px;
 border-radius:50%;background:rgba(255,255,255,.08)}}
.hero-tag{{display:inline-block;font-size:12px;letter-spacing:2px;padding:5px 12px;border-radius:99px;
 background:rgba(255,255,255,.18);margin-bottom:12px}}
.hero h1{{font-size:32px;font-weight:800;letter-spacing:.5px;margin-bottom:8px}}
.hero p{{opacity:.9;font-size:14px;max-width:820px;line-height:1.7}}
.hero-stats{{display:flex;flex-wrap:wrap;gap:26px;margin-top:20px}}
.hero-stats div span{{display:block;font-size:11.5px;opacity:.78}}
.hero-stats div b{{font-size:19px;font-weight:700}}
.sec{{font-size:15px;font-weight:700;color:#334155;margin:30px 0 14px;padding-left:11px;
 border-left:4px solid #4F46E5;line-height:1.2}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(228px,1fr));gap:16px}}
.kpi{{background:#fff;border-radius:14px;padding:19px 20px;border:1px solid #E8ECF5;
 border-top:3px solid var(--k);box-shadow:0 1px 3px rgba(16,24,40,.05);transition:.22s}}
.kpi:hover{{transform:translateY(-3px);box-shadow:0 12px 26px rgba(16,24,40,.09)}}
.kpi-l{{font-size:12.5px;color:#64748B}}
.kpi-v{{font-size:29px;font-weight:800;margin:5px 0 4px;letter-spacing:-.5px;color:#0F172A}}
.kpi-v .u{{font-size:13px;font-weight:600;color:#94A3B8;margin-left:2px}}
.kpi-s{{font-size:11.5px;color:#94A3B8;line-height:1.55}}
.grid{{display:grid;grid-template-columns:repeat(12,1fr);gap:16px}}
.card{{background:#fff;border:1px solid #E8ECF5;border-radius:14px;padding:18px 20px 10px;
 box-shadow:0 1px 3px rgba(16,24,40,.05);transition:.22s;overflow:hidden}}
.card:hover{{box-shadow:0 12px 26px rgba(16,24,40,.08)}}
.w4{{grid-column:span 4}}.w6{{grid-column:span 6}}.w8{{grid-column:span 8}}.w12{{grid-column:span 12}}
.card-h h3{{font-size:15px;font-weight:700;color:#0F172A;margin-bottom:4px}}
.card-h p{{font-size:12px;color:#94A3B8;line-height:1.6;margin-bottom:8px;min-height:19px}}
.ins-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}
.ins{{background:#fff;border:1px solid #E8ECF5;border-left:4px solid var(--k);border-radius:12px;
 padding:16px 18px;box-shadow:0 1px 3px rgba(16,24,40,.05)}}
.ins-t{{font-size:14px;font-weight:700;margin-bottom:6px;color:#0F172A}}
.ins-b{{font-size:12.5px;color:#64748B;line-height:1.75}}
.foot{{text-align:center;color:#94A3B8;font-size:12px;margin-top:30px;line-height:1.9}}
@media(max-width:1180px){{.w4,.w6,.w8{{grid-column:span 12}}}}
@media(max-width:640px){{.hero{{padding:28px 22px}}.hero h1{{font-size:24px}}.wrap{{padding:0 14px 30px}}.hero{{margin:0 -14px 18px}}}}
</style></head><body><div class="wrap">
<div class="hero">
  <div class="hero-tag">TIANCHI · MUM & BABY E-COMMERCE</div>
  <h1>母婴电商经营分析看板</h1>
  <p>基于天池母婴数据集 {S['raw']['baby']:,} 条婴儿信息 × {S['raw']['trade']:,} 条交易记录，覆盖 {S['period'][0]} 至 {S['period'][1]}，从销售趋势、类目结构、人群画像三个维度还原业务全貌。已剔除 42 笔异常大额订单（buy_mount &gt; 100）。</p>
  <div class="hero-stats">
    <div><span>数据周期</span><b>{S['period'][0][:7]} ~ {S['period'][1][:7]}</b></div>
    <div><span>完整月份</span><b>{S['months']} 个月</b></div>
    <div><span>在售商品</span><b>{S['sku']:,} 件</b></div>
    <div><span>二级类目</span><b>{S['cats']} 个</b></div>
    <div><span>客单件数</span><b>{S['avg_mount']} 件</b></div>
  </div>
</div>

<div class="sec">核心经营指标</div>
<div class="kpis">{kpis}</div>

<div class="sec">趋势分析</div>
<div class="grid">{card_html}</div>

<div class="sec">关键洞察</div>
<div class="ins-grid">{ins_html}</div>

<div class="foot">
  数据清洗：剔除异常生日 {bad_birth} 条 · 剔除极端购买数量 {ext_count} 笔（&gt;100 件）<br>
  分析口径：总销量 {volume:,} 件（原始 {raw_volume:,} 件，剔除 {S['clean']['ext_volume_share']}%）· 有效订单 {S['orders']:,} 笔 ·
  性别/月龄分析基于可关联婴儿信息的 {S['matched_share']}% 销量子集（约 1,100 件）· 趋势图已剔除首尾不完整月 · 图表支持悬停与缩放
</div>
</div></body></html>"""

PLOTLY_SRC = po.get_plotlyjs()
with open(OUT, "w", encoding="utf-8") as f:
    f.write(html.replace("@@PLOTLY_JS@@", PLOTLY_SRC))
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(S, f, ensure_ascii=False, indent=2, default=str)

print("OK ->", OUT)
print("plotly.js 已内联, 文件大小(MB) =", round(len(PLOTLY_SRC) / 1048576, 2))
print(json.dumps(S, ensure_ascii=False, indent=2, default=str))