from src.data_prep import generate_synthetic_reviews

def test_generator_shapes_and_columns():
    df = generate_synthetic_reviews(n=100, seed=1)
    assert len(df) == 100
    required = {
        "review_text",
        "rating_overall",
        "rating_seat",
        "rating_food",
        "rating_cabin_service",
        "rating_punctuality",
        "prior_trips_12m",
        "advance_purchase_days",
        "route",
        "cabin",
        "loyalty_tier",
        "perk_lounge_voucher",
        "purchase_intent",
    }
    assert required.issubset(df.columns)

def test_purchase_intent_is_binary():
    df = generate_synthetic_reviews(n=200, seed=42)
    assert set(df["purchase_intent"].unique()).issubset({0, 1})

def test_value_ranges():
    df = generate_synthetic_reviews(n=50, seed=7)
    for col in [
        "rating_overall",
        "rating_seat",
        "rating_food",
        "rating_cabin_service",
        "rating_punctuality",
    ]:
        assert float(df[col].min()) >= 1.0
        assert float(df[col].max()) <= 5.0
    assert int(df["advance_purchase_days"].min()) >= 0
