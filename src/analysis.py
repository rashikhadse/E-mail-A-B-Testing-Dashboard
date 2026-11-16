"""Metric and A/B-test helpers (pure functions, no Streamlit)."""
from dataclasses import dataclass
from math import ceil, sqrt

import numpy as np
import pandas as pd
from scipy.stats import norm

# metric -> (numerator column, denominator definition)
METRICS = {
    "Open rate": "opened",
    "Reply rate": "replied",
    "Click rate": "clicked",
    "Bounce rate": "bounced",
}


def _counts(df: pd.DataFrame, metric: str) -> tuple[int, int]:
    """Return (successes, trials). Bounce uses all sends; the rest use delivered emails."""
    col = METRICS[metric]
    if metric == "Bounce rate":
        return int(df[col].sum()), int(df["sent"].sum())
    delivered = df[df["bounced"] == 0]
    return int(delivered[col].sum()), int(len(delivered))


def rate(df: pd.DataFrame, metric: str) -> float:
    x, n = _counts(df, metric)
    return x / n if n else float("nan")


def kpis(df: pd.DataFrame) -> dict:
    return {"Emails sent": int(df["sent"].sum()), **{m: rate(df, m) for m in METRICS}}


def rate_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """All four rates grouped by a column (e.g. persona, industry, send_window)."""
    rows = []
    for key, g in df.groupby(by):
        rows.append({by: key, "sent": int(g["sent"].sum()),
                     **{m: rate(g, m) for m in METRICS}})
    return pd.DataFrame(rows).sort_values("Open rate", ascending=False).reset_index(drop=True)


@dataclass
class ZTestResult:
    p1: float
    p2: float
    n1: int
    n2: int
    diff: float          # p2 - p1 (absolute)
    lift: float          # relative lift of p2 over p1
    z: float
    p_value: float       # two-sided
    ci_low: float        # CI for the absolute difference
    ci_high: float
    significant: bool
    alpha: float


def two_proportion_ztest(x1: int, n1: int, x2: int, n2: int, alpha: float = 0.05) -> ZTestResult:
    """Two-sided two-proportion z-test, pooled SE for the test, unpooled SE for the CI.

    H0: p1 == p2.  Variant 1 is the control (A), variant 2 the challenger (B).
    """
    if min(n1, n2) <= 0:
        raise ValueError("Both groups need at least one observation.")
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se_pooled = sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se_pooled if se_pooled > 0 else 0.0
    p_value = 2 * (1 - norm.cdf(abs(z)))
    se_unpooled = sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    zcrit = norm.ppf(1 - alpha / 2)
    diff = p2 - p1
    return ZTestResult(
        p1=p1, p2=p2, n1=n1, n2=n2, diff=diff,
        lift=diff / p1 if p1 > 0 else float("nan"),
        z=z, p_value=p_value,
        ci_low=diff - zcrit * se_unpooled, ci_high=diff + zcrit * se_unpooled,
        significant=p_value < alpha, alpha=alpha,
    )


def ab_test(df: pd.DataFrame, metric: str, alpha: float = 0.05) -> ZTestResult:
    """Run the A vs B test for one metric on a dataframe holding both variants."""
    a, b = df[df["variant"] == "A"], df[df["variant"] == "B"]
    x1, n1 = _counts(a, metric)
    x2, n2 = _counts(b, metric)
    return two_proportion_ztest(x1, n1, x2, n2, alpha)


def ab_test_all_campaigns(df: pd.DataFrame, metric: str, alpha: float = 0.05) -> pd.DataFrame:
    """One row per campaign. Applies a Bonferroni-adjusted threshold across campaigns."""
    camps = sorted(df["campaign"].unique())
    adj_alpha = alpha / len(camps)
    rows = []
    for c in camps:
        r = ab_test(df[df["campaign"] == c], metric, alpha)
        rows.append({
            "campaign": c, "A rate": r.p1, "B rate": r.p2, "abs diff": r.diff,
            "relative lift": r.lift, "z": r.z, "p-value": r.p_value,
            "significant (alpha)": r.significant,
            "significant (Bonferroni)": r.p_value < adj_alpha,
        })
    return pd.DataFrame(rows)


def required_sample_size(baseline: float, mde_abs: float, alpha: float = 0.05,
                         power: float = 0.80) -> int:
    """Emails needed PER VARIANT to detect an absolute lift of `mde_abs`."""
    if not 0 < baseline < 1 or mde_abs <= 0 or baseline + mde_abs >= 1:
        raise ValueError("Need 0 < baseline and baseline + mde < 1, with mde > 0.")
    p1, p2 = baseline, baseline + mde_abs
    pbar = (p1 + p2) / 2
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    num = (za * sqrt(2 * pbar * (1 - pbar)) + zb * sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return ceil(num / mde_abs ** 2)
