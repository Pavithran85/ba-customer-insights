import joblib, yaml
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from scipy import sparse

def build_features_for_inference(df: pd.DataFrame, vec, expected_feature_names):
    # 1) Text from saved vectorizer (NO refit)
    X_text = vec.transform(df["review_text"].fillna(""))
    n_text = len(vec.get_feature_names_out())

    # 2) Numeric & categorical (same cols/order as training)
    num = df[[
        "rating_overall","rating_seat","rating_food","rating_cabin_service",
        "rating_punctuality","prior_trips_12m","advance_purchase_days"
    ]].apply(pd.to_numeric, errors="coerce").fillna(0)

    cat = pd.get_dummies(df[["route","cabin","loyalty_tier"]], drop_first=True)

    tab = pd.concat([num, cat], axis=1)

    # Align to the exact tabular feature order the model expects
    expected_tabular = expected_feature_names[n_text:]
    for col in expected_tabular:
        if col not in tab.columns:
            tab[col] = 0
    tab = tab[expected_tabular].apply(pd.to_numeric, errors="coerce").fillna(0).astype("float64")

    X = sparse.hstack([X_text, sparse.csr_matrix(tab.values, dtype="float64")], format="csr")
    return X

def main(cfg_path="configs/default.yaml"):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    paths = cfg["paths"]; target = cfg["model"]["target"]

    # Load data and trained bundle
    df = pd.read_csv(f'{paths["processed"]}/dataset.csv')
    bundle = joblib.load(f'{paths["models"]}/ba_buying_xgb.joblib')
    model, vec, feat_names = bundle["model"], bundle["vectorizer"], bundle["features"]

    # Build features using saved vectorizer + aligned columns
    X = build_features_for_inference(df, vec, feat_names)
    y = df[target].values
    y_prob = model.predict_proba(X)[:, 1]

    out = Path(paths["reports"]); out.mkdir(parents=True, exist_ok=True)

    # ROC / PR curves
    RocCurveDisplay.from_predictions(y, y_prob)
    plt.savefig(out/"roc_curve.png", dpi=160, bbox_inches="tight"); plt.close()
    PrecisionRecallDisplay.from_predictions(y, y_prob)
    plt.savefig(out/"pr_curve.png", dpi=160, bbox_inches="tight"); plt.close()

    # SHAP (tree explainer)
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X, check_additivity=False)
    shap.summary_plot(shap_values, show=False, max_display=15, feature_names=feat_names)
    plt.savefig(out/"shap_summary.png", dpi=160, bbox_inches="tight"); plt.close()

    print("✅ Evaluation complete. Plots saved in models/reports/")

if __name__ == "__main__":
    main()
