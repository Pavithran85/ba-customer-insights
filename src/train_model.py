import yaml, joblib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from .data_prep import generate_synthetic_reviews, save_csv
from .features import build_features

def main(cfg_path="configs/default.yaml"):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    paths = cfg["paths"]; target = cfg["model"]["target"]

    df = generate_synthetic_reviews(n=8000)
    save_csv(df, f'{paths["processed"]}/dataset.csv')

    X, feat_names, vec = build_features(df)
    y = df[target].values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    params = cfg["model"]["params"]
    clf = XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        objective="binary:logistic",
        eval_metric="auc"
    )
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, proba)
    ap  = average_precision_score(yte, proba)

    Path(paths["models"]).mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "vectorizer": vec, "features": feat_names},
                f'{paths["models"]}/ba_buying_xgb.joblib')

    report = Path(paths["reports"]); report.mkdir(parents=True, exist_ok=True)
    Path(report/"metrics.txt").write_text(f"AUC: {auc:.3f}\nPR-AUC: {ap:.3f}\n")
    print(f"Saved model. AUC={auc:.3f}, PR-AUC={ap:.3f}")

if __name__ == "__main__":
    main()
