import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def build_features(df: pd.DataFrame, text_col="review_text", max_features=5000):
    vec = TfidfVectorizer(min_df=5, max_df=0.8, ngram_range=(1,2))
    X_text = vec.fit_transform(df[text_col].fillna(""))
    cat = pd.get_dummies(df[["route","cabin","loyalty_tier"]], drop_first=True)
    num = df[["rating_overall","rating_seat","rating_food","rating_cabin_service",
              "rating_punctuality","prior_trips_12m","advance_purchase_days"]].fillna(0)
    from scipy import sparse
    X = sparse.hstack([X_text, sparse.csr_matrix(num.values), sparse.csr_matrix(cat.values)])
    feature_names = list(vec.get_feature_names_out()) + list(num.columns) + list(cat.columns)
    return X, feature_names, vec
