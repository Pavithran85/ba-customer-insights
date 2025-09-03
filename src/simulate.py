import yaml, joblib
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import sparse

def apply_changes(df, change_dict):
    df = df.copy()
    for k, v in change_dict.items():
        if k in df.columns:
            if df[k].dtype.kind in "if":
                df[k] = np.clip(df[k] + v, 1, 5) if "rating" in k else df[k] + v
            else:
                df[k] = v
        else:
            df[k] = v
    return df

def build_features_for_inference(df: pd.DataFrame, vec, expected_feature_names):
    # 1) Text from saved vectorizer
    X_text = vec.transform(df["review_text"].fillna(""))
    n_text = len(vec.get_feature_names_out())

    # 2) Numeric & categorical (same as training)
    num = df[[
        "rating_overall","rating_seat","rating_food","rating_cabin_service",
        "rating_punctuality","prior_trips_12m","advance_purchase_days"
    ]].apply(pd.to_numeric, errors="coerce").fillna(0)

    cat = pd.get_dummies(df[["route","cabin","loyalty_tier"]], drop_first=True)

    tab = pd.concat([num, cat], axis=1)

    expected_tabular = expected_feature_names[n_text:]
    for col in expected_tabular:
        if col not in tab.columns:
            tab[col] = 0
    tab = tab[expected_tabular]

    # 🔧 force numeric float dtype for sparse matrix
    tab = tab.apply(pd.to_numeric, errors="coerce").fillna(0).astype("float64")

    from scipy import sparse
    X = sparse.hstack([X_text, sparse.csr_matrix(tab.values, dtype="float64")], format="csr")
    return X

def score(df, bundle):
    model = bundle["model"]
    vec = bundle["vectorizer"]
    feature_names = bundle["features"]  # order the model was trained on
    X = build_features_for_inference(df, vec, feature_names)
    return model.predict_proba(X)[:, 1]

def main(cfg_path="configs/default.yaml"):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    paths = cfg["paths"]; sims = cfg["simulation"]["scenarios"]

    df = pd.read_csv(f'{paths["processed"]}/dataset.csv')
    bundle = joblib.load(f'{paths["models"]}/ba_buying_xgb.joblib')

    base_p = score(df, bundle).mean()

    rows = []
    for sc in sims:
        changed = apply_changes(df, sc["change"])
        p = score(changed, bundle).mean()
        rows.append({
            "scenario": sc["name"],
            "baseline": base_p,
            "scenario_p": p,
            "avg_lift": p - base_p
        })

    out = pd.DataFrame(rows)
    Path(paths["reports"]).mkdir(parents=True, exist_ok=True)
    out.to_csv(f'{paths["reports"]}/simulations.csv', index=False)
    print(out)

if __name__ == "__main__":
    main()
