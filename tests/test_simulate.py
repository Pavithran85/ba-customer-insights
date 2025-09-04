import pandas as pd
from src.simulate import apply_changes

def test_apply_changes_numeric_clip():
    df = pd.DataFrame({"rating_seat": [4.8], "rating_punctuality": [4.9]})
    out = apply_changes(df, {"rating_seat": 1.0, "rating_punctuality": 1.0})
    # ratings are clipped to max 5
    assert float(out.loc[0, "rating_seat"]) == 5.0
    assert float(out.loc[0, "rating_punctuality"]) == 5.0

def test_apply_changes_adds_new_flag_column():
    df = pd.DataFrame({"rating_seat": [4.0]})
    out = apply_changes(df, {"perk_lounge_voucher": 1})
    assert "perk_lounge_voucher" in out.columns
    assert int(out.loc[0, "perk_lounge_voucher"]) == 1

def test_apply_changes_overwrites_non_numeric():
    df = pd.DataFrame({"cabin": ["Economy"]})
    out = apply_changes(df, {"cabin": "Business"})
    assert out.loc[0, "cabin"] == "Business"
