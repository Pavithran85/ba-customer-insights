import numpy as np, pandas as pd
from pathlib import Path

def generate_synthetic_reviews(n=5000, seed=7):
    rng = np.random.default_rng(seed)
    cabins = ["Economy","Premium Economy","Business","First"]
    routes = ["LHR-JFK","LHR-DOH","LHR-SIN","LGW-AGP","MAN-AMS"]
    tiers  = ["None","Blue","Bronze","Silver","Gold"]

    rating_overall = np.clip(rng.normal(3.8, 0.9, n), 1, 5)
    rating_seat    = np.clip(rng.normal(3.7, 1.0, n), 1, 5)
    rating_food    = np.clip(rng.normal(3.5, 1.0, n), 1, 5)
    rating_cabin   = np.clip(rng.normal(4.0, 0.8, n), 1, 5)
    rating_punct   = np.clip(rng.normal(4.1, 0.7, n), 1, 5)

    df = pd.DataFrame({
        "review_text": np.where(rating_overall>4.2,
            "Great crew and smooth flight. Comfortable seats.",
            np.where(rating_overall<3.0,
            "Delays and cramped seats. Food could be better.",
            "Decent experience; some room for improvement.")),
        "rating_overall": rating_overall,
        "rating_seat": rating_seat,
        "rating_food": rating_food,
        "rating_cabin_service": rating_cabin,
        "rating_punctuality": rating_punct,
        "prior_trips_12m": rng.poisson(2, n),
        "advance_purchase_days": np.clip(rng.normal(28, 15, n), 0, 120).astype(int),
        "route": rng.choice(routes, n, p=[.35,.15,.2,.2,.1]),
        "cabin": rng.choice(cabins, n, p=[.65,.15,.18,.02]),
        "loyalty_tier": rng.choice(tiers, n, p=[.45,.2,.15,.12,.08]),
        "perk_lounge_voucher": 0
    })

    logit = (
        -1.0
        + 0.5*df["rating_overall"]
        + 0.2*df["rating_punctuality"]
        + 0.15*df["prior_trips_12m"]
        + 0.01*df["advance_purchase_days"]
        + df["cabin"].map({"Economy":0,"Premium Economy":0.1,"Business":0.4,"First":0.5})
        + df["loyalty_tier"].map({"None":0,"Blue":0.05,"Bronze":0.1,"Silver":0.2,"Gold":0.35})
    )
    p = 1/(1+np.exp(-logit))
    df["purchase_intent"] = (rng.random(n) < p).astype(int)
    return df

def save_csv(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
