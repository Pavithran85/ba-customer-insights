import yaml, joblib, pandas as pd
from pathlib import Path
from src.evaluate import build_features_for_inference

def test_build_features_for_inference_shape_matches_model_bundle():
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
    paths = cfg["paths"]
    df = pd.read_csv(f'{paths["processed"]}/dataset.csv')
    bundle = joblib.load(f'{paths["models"]}/ba_buying_xgb.joblib')
    vec, feat_names = bundle["vectorizer"], bundle["features"]
    X = build_features_for_inference(df, vec, feat_names)
    # should match the # of features model was trained with
    assert X.shape[1] == len(feat_names)
