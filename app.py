import yaml, joblib, pandas as pd
import streamlit as st
from src.simulate import apply_changes
from src.evaluate import build_features_for_inference  # reuse helper

st.set_page_config(page_title="BA Customer Insights", layout="wide")
st.title("British Airways – Customer Insights & Uplift Simulator")

cfg = yaml.safe_load(open("configs/default.yaml").read())
paths = cfg["paths"]
df = pd.read_csv(f'{paths["processed"]}/dataset.csv')
bundle = joblib.load(f'{paths["models"]}/ba_buying_xgb.joblib')
model, vec, feat_names = bundle["model"], bundle["vectorizer"], bundle["features"]

# Baseline score
X_base = build_features_for_inference(df, vec, feat_names)
base = model.predict_proba(X_base)[:,1].mean()

st.sidebar.header("Scenario Controls")
seat = st.sidebar.slider("Seat comfort +", 0.0, 1.5, 0.5, 0.1)
otp  = st.sidebar.slider("Punctuality +", 0.0, 1.5, 1.0, 0.1)
lounge = st.sidebar.checkbox("Lounge voucher")

changes = {"rating_seat": seat, "rating_punctuality": otp, "perk_lounge_voucher": int(lounge)}
df2 = apply_changes(df, changes)
X2 = build_features_for_inference(df2, vec, feat_names)
scen = model.predict_proba(X2)[:,1].mean()

c1, c2, c3 = st.columns(3)
c1.metric("Baseline conversion (avg prob)", f"{base:.3f}")
c2.metric("Scenario conversion", f"{scen:.3f}")
c3.metric("Avg uplift", f"{(scen-base):+.3f}")

st.subheader("Sample Reviews")
st.dataframe(df[["route","cabin","rating_overall","rating_punctuality","review_text"]].head(20))
