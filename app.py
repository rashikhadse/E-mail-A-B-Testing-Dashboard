"""Cold Email A/B Testing & Campaign Analytics Dashboard.  Run: streamlit run app.py"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.analysis import (METRICS, ab_test, ab_test_all_campaigns, kpis, rate_table,
                          required_sample_size)

st.set_page_config(page_title="Cold Email Analytics", page_icon="📧", layout="wide")
DATA = Path(__file__).parent / "data" / "campaigns.csv"


@st.cache_data
def load(path) -> pd.DataFrame:
    return pd.read_csv(path)


def fmt(x: float) -> str:
    return f"{x:.1%}"


st.title("📧 Cold Email A/B Testing & Campaign Analytics")
st.caption("Simulated B2B outbound data. Open, click and reply rates use delivered emails as the denominator; bounce rate uses emails sent.")

# ---------- data source ----------
with st.sidebar:
    st.header("Data")
    up = st.file_uploader("Upload your own CSV (optional)", type="csv",
                          help="Needs columns: campaign, variant (A/B), persona, industry, send_window, sent, bounced, opened, clicked, replied")
if up is not None:
    df = pd.read_csv(up)
    need = {"campaign", "variant", "persona", "industry", "send_window", "sent",
            "bounced", "opened", "clicked", "replied"}
    if not need <= set(df.columns):
        st.error(f"Missing columns: {sorted(need - set(df.columns))}")
        st.stop()
elif DATA.exists():
    df = load(DATA)
else:
    st.error("data/campaigns.csv not found. Run `python -m src.generate_data` first.")
    st.stop()

# ---------- filters ----------
with st.sidebar:
    st.header("Filters")
    personas = st.multiselect("Persona", sorted(df["persona"].unique()), default=sorted(df["persona"].unique()))
    industries = st.multiselect("Industry", sorted(df["industry"].unique()), default=sorted(df["industry"].unique()))
    windows = st.multiselect("Send window", sorted(df["send_window"].unique()), default=sorted(df["send_window"].unique()))
fdf = df[df["persona"].isin(personas) & df["industry"].isin(industries) & df["send_window"].isin(windows)]
if fdf.empty:
    st.warning("No rows match the current filters.")
    st.stop()

tab_over, tab_ab, tab_seg, tab_size, tab_data = st.tabs(
    ["Overview", "A/B test", "Segments", "Sample size", "Data"])

# ---------- overview ----------
with tab_over:
    k = kpis(fdf)
    cols = st.columns(5)
    cols[0].metric("Emails sent", f"{k['Emails sent']:,}")
    for col, m in zip(cols[1:], METRICS):
        col.metric(m, fmt(k[m]))
    st.subheader("Campaign performance")
    rows = []
    for c, g in fdf.groupby("campaign"):
        rows.append({"campaign": c, "sent": int(g["sent"].sum()), **{m: kpis(g)[m] for m in METRICS}})
    ct = pd.DataFrame(rows)
    st.dataframe(ct.style.format({m: "{:.1%}" for m in METRICS}), width="stretch", hide_index=True)
    funnel = pd.DataFrame({
        "stage": ["Sent", "Delivered", "Opened", "Clicked", "Replied"],
        "emails": [int(fdf["sent"].sum()), int((fdf["bounced"] == 0).sum()),
                   int(fdf["opened"].sum()), int(fdf["clicked"].sum()), int(fdf["replied"].sum())]})
    st.subheader("Funnel")
    st.altair_chart(alt.Chart(funnel).mark_bar().encode(
        x=alt.X("emails:Q"), y=alt.Y("stage:N", sort=None),
        tooltip=["stage", "emails"]).properties(height=220), width="stretch")

# ---------- A/B ----------
with tab_ab:
    c1, c2, c3 = st.columns(3)
    camp = c1.selectbox("Campaign", sorted(fdf["campaign"].unique()))
    metric = c2.selectbox("Metric", list(METRICS), index=0)
    alpha = c3.select_slider("Significance level (alpha)", [0.01, 0.05, 0.10], value=0.05)
    cdf = fdf[fdf["campaign"] == camp]
    if cdf["variant"].nunique() < 2:
        st.warning("Both variants need data. Loosen the filters.")
    else:
        subj = cdf.groupby("variant")["subject_line"].first() if "subject_line" in cdf else {}
        if len(subj):
            st.markdown(f"**A:** {subj.get('A', '')}  \n**B:** {subj.get('B', '')}")
        r = ab_test(cdf, metric, alpha)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Variant A", fmt(r.p1), help=f"n = {r.n1:,}")
        m2.metric("Variant B", fmt(r.p2), f"{r.diff * 100:+.2f} pts")
        m3.metric("Relative lift", f"{r.lift:+.1%}")
        m4.metric("p-value", f"{r.p_value:.4f}")
        if r.significant:
            winner = "B" if r.diff > 0 else "A"
            st.success(f"Statistically significant at alpha = {alpha}: variant {winner} wins "
                       f"(z = {r.z:.2f}). 95% CI for B minus A: [{r.ci_low * 100:+.2f}, {r.ci_high * 100:+.2f}] pts.")
        else:
            st.info(f"No statistically significant difference at alpha = {alpha} (z = {r.z:.2f}). "
                    f"The CI for B minus A, [{r.ci_low * 100:+.2f}, {r.ci_high * 100:+.2f}] pts, includes 0. "
                    "Keep testing or collect more sends before deciding.")
        plot = pd.DataFrame({"variant": ["A", "B"], "rate": [r.p1, r.p2]})
        st.altair_chart(alt.Chart(plot).mark_bar().encode(
            x="variant:N", y=alt.Y("rate:Q", axis=alt.Axis(format="%")),
            tooltip=["variant", alt.Tooltip("rate:Q", format=".2%")]).properties(height=260),
            width="stretch")
    st.subheader(f"All campaigns: {metric}")
    allc = ab_test_all_campaigns(fdf, metric, alpha)
    st.dataframe(allc.style.format({"A rate": "{:.1%}", "B rate": "{:.1%}", "abs diff": "{:+.2%}",
                                    "relative lift": "{:+.1%}", "z": "{:.2f}", "p-value": "{:.4f}"}),
                 width="stretch", hide_index=True)
    st.caption("Testing several campaigns at once inflates false positives, so the Bonferroni column uses alpha divided by the number of campaigns.")

# ---------- segments ----------
with tab_seg:
    dim = st.radio("Break down by", ["persona", "industry", "send_window"], horizontal=True)
    seg_metric = st.selectbox("Metric", list(METRICS), key="segm")
    t = rate_table(fdf, dim)
    st.altair_chart(alt.Chart(t).mark_bar().encode(
        x=alt.X(f"{seg_metric}:Q", axis=alt.Axis(format="%")), y=alt.Y(f"{dim}:N", sort="-x"),
        tooltip=[dim, "sent", alt.Tooltip(f"{seg_metric}:Q", format=".2%")]).properties(height=300),
        width="stretch")
    st.dataframe(t.style.format({m: "{:.1%}" for m in METRICS}), width="stretch", hide_index=True)
    st.subheader("Persona x send window heatmap (open rate)")
    sub = fdf[fdf["bounced"] == 0]
    heat = sub.groupby(["persona", "send_window"])["opened"].mean().reset_index()
    st.altair_chart(alt.Chart(heat).mark_rect().encode(
        x="send_window:N", y="persona:N",
        color=alt.Color("opened:Q", title="Open rate", scale=alt.Scale(scheme="blues")),
        tooltip=["persona", "send_window", alt.Tooltip("opened:Q", format=".1%")]).properties(height=260),
        width="stretch")

# ---------- sample size ----------
with tab_size:
    st.markdown("How many emails per variant do you need before a test can detect a given lift?")
    a, b, c = st.columns(3)
    base = a.slider("Baseline rate", 0.01, 0.80, 0.40, 0.01, format="%.2f")
    mde = b.slider("Smallest lift to detect (pts)", 0.01, 0.20, 0.05, 0.01, format="%.2f")
    pw = c.select_slider("Power", [0.7, 0.8, 0.9], value=0.8)
    try:
        n = required_sample_size(base, mde, 0.05, pw)
        st.metric("Emails needed per variant", f"{n:,}", help="Total sends = 2 x this number")
    except ValueError as e:
        st.error(str(e))

# ---------- data ----------
with tab_data:
    st.dataframe(fdf.head(1000), width="stretch")
    st.download_button("Download filtered data (CSV)", fdf.to_csv(index=False), "filtered_campaigns.csv", "text/csv")
