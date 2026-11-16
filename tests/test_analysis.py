import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

from src.analysis import (ab_test, kpis, rate, rate_table, required_sample_size,
                          two_proportion_ztest)
from src.generate_data import generate


def test_ztest_matches_chi_square_without_correction():
    """Pooled two-sided z-test p-value equals the uncorrected chi-square p-value."""
    x1, n1, x2, n2 = 380, 1000, 460, 1000
    r = two_proportion_ztest(x1, n1, x2, n2)
    table = [[x1, n1 - x1], [x2, n2 - x2]]
    _, p, _, _ = chi2_contingency(table, correction=False)
    assert r.p_value == pytest.approx(p, rel=1e-6)
    assert r.z == pytest.approx(np.sqrt(chi2_contingency(table, correction=False)[0]), rel=1e-6)


def test_no_difference_is_not_significant():
    r = two_proportion_ztest(100, 1000, 100, 1000)
    assert r.z == 0 and r.p_value == pytest.approx(1.0) and not r.significant


def test_ci_contains_difference_and_sign():
    r = two_proportion_ztest(300, 1000, 350, 1000)
    assert r.ci_low < r.diff < r.ci_high
    assert r.diff == pytest.approx(0.05)
    assert r.lift == pytest.approx(0.05 / 0.30)


def test_invalid_input():
    with pytest.raises(ValueError):
        two_proportion_ztest(1, 0, 1, 10)


def test_sample_size_known_value():
    # 20% -> 25% at 80% power, alpha .05 needs ~1,090-1,100 per arm
    assert 1050 <= required_sample_size(0.20, 0.05) <= 1150
    with pytest.raises(ValueError):
        required_sample_size(0.99, 0.05)


def test_rate_denominators():
    df = pd.DataFrame({"sent": [1, 1, 1, 1], "bounced": [1, 0, 0, 0],
                       "opened": [0, 1, 1, 0], "clicked": [0, 1, 0, 0],
                       "replied": [0, 0, 1, 0], "variant": list("AABB")})
    assert rate(df, "Bounce rate") == pytest.approx(0.25)   # of sent
    assert rate(df, "Open rate") == pytest.approx(2 / 3)    # of delivered
    assert kpis(df)["Emails sent"] == 4


def test_simulated_data_integrity():
    df = generate(n_per_variant=500, seed=1)
    assert (df["opened"] <= 1 - df["bounced"]).all()        # bounced emails never open
    assert (df["clicked"] <= df["opened"]).all()
    assert (df["replied"] <= df["opened"]).all()
    assert df["lead_id"].is_unique


def test_planted_effect_is_detected():
    df = generate(n_per_variant=3000, seed=7)
    c01 = df[df["campaign"] == "C01 Hiring speed"]
    assert ab_test(c01, "Open rate").significant          # planted +8pt
    c02 = df[df["campaign"] == "C02 Cost per hire"]
    assert not ab_test(c02, "Open rate", alpha=0.001).significant  # ~+1pt, tiny


def test_rate_table_columns():
    df = generate(n_per_variant=200, seed=3)
    t = rate_table(df, "persona")
    assert {"persona", "sent", "Open rate", "Reply rate"} <= set(t.columns)
