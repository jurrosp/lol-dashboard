import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


FEATURES = [
    "kp",
    "dpm",
    "gpm",
    "team_dmg_pct",
    "vision_score",
    "deaths_per_10",
]


def train_win_model(df: pd.DataFrame):
    """
    Train logistic regression model om win/loss te voorspellen.
    """
    if len(df) < 25:
        return None

    df = df.dropna(subset=["win"]).copy()

    X = df[FEATURES].fillna(0)
    y = df["win"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    weights = pd.Series(model.coef_[0], index=FEATURES)
    weights = weights.sort_values(key=lambda s: s.abs(), ascending=False)

    return {
        "auc": auc,
        "weights": weights,
    }
