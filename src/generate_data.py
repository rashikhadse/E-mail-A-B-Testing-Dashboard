"""Simulate a B2B cold-email programme.

Every row is one email sent to one contact. Real effects are planted so the
dashboard has something to find (and the A/B test has something to detect):

* Variant B subject lines (question style) get higher open rates in some campaigns.
* Tuesday/Thursday morning sends open better than Friday afternoon.
* Persona and industry shift baseline engagement.
* Bounces depend on a per-campaign list quality factor.
* Replies and clicks can only happen after an open.

The data is SIMULATED. Say so wherever you present results.
Run:  python -m src.generate_data
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
PERSONAS = {  # persona: open-rate multiplier
    "HR Head": 1.10, "Talent Acquisition Manager": 1.15, "Founder / CEO": 0.90,
    "CTO / VP Engineering": 0.95, "Recruitment Consultant": 1.05,
}
INDUSTRIES = {"SaaS": 1.05, "Fintech": 1.00, "E-commerce": 0.95,
              "Healthcare": 0.90, "Manufacturing": 0.85, "EdTech": 1.00}
SEND_WINDOWS = {  # window: open-rate multiplier
    "Tue 9-11 AM": 1.15, "Tue 2-4 PM": 1.00, "Thu 9-11 AM": 1.10,
    "Thu 2-4 PM": 0.95, "Fri 3-5 PM": 0.80,
}
# campaign -> (subject A, subject B, A open rate, B open rate, list-quality bounce rate)
CAMPAIGNS = {
    "C01 Hiring speed": ("Faster hiring with AI", "Struggling to fill roles quickly?", 0.38, 0.46, 0.035),
    "C02 Cost per hire": ("Cut your cost per hire", "Cost per hire down 30%?", 0.40, 0.41, 0.040),
    "C03 Candidate quality": ("Better candidates, less screening", "How do you screen 500 resumes?", 0.35, 0.44, 0.030),
    "C04 Demo invite": ("Quick demo of AI candidate sourcing", "15 minutes this week?", 0.42, 0.45, 0.060),
    "C05 Case study": ("How a startup hired 10 engineers", "Want to hire 10 engineers in a month?", 0.39, 0.37, 0.045),
}
CLICK_GIVEN_OPEN = 0.16
REPLY_GIVEN_OPEN = {"A": 0.075, "B": 0.095}


def generate(n_per_variant: int = 1500, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    cid = 0
    for camp, (subj_a, subj_b, open_a, open_b, bounce) in CAMPAIGNS.items():
        for variant, subject, base_open in (("A", subj_a, open_a), ("B", subj_b, open_b)):
            n = n_per_variant
            persona = rng.choice(list(PERSONAS), n, p=[.22, .26, .20, .16, .16])
            industry = rng.choice(list(INDUSTRIES), n)
            window = rng.choice(list(SEND_WINDOWS), n, p=[.25, .20, .25, .15, .15])
            p_open = np.clip(
                base_open
                * np.array([PERSONAS[p] for p in persona])
                * np.array([INDUSTRIES[i] for i in industry])
                * np.array([SEND_WINDOWS[w] for w in window]),
                0.02, 0.95,
            )
            bounced = rng.random(n) < bounce
            opened = (~bounced) & (rng.random(n) < p_open)
            clicked = opened & (rng.random(n) < CLICK_GIVEN_OPEN)
            replied = opened & (rng.random(n) < REPLY_GIVEN_OPEN[variant])
            for k in range(n):
                cid += 1
                rows.append((f"L{cid:06d}", camp, variant, subject, persona[k],
                             industry[k], window[k], 1, int(bounced[k]),
                             int(opened[k]), int(clicked[k]), int(replied[k])))
    return pd.DataFrame(rows, columns=[
        "lead_id", "campaign", "variant", "subject_line", "persona", "industry",
        "send_window", "sent", "bounced", "opened", "clicked", "replied"])


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "campaigns.csv"
    df = generate()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows to {out}")
