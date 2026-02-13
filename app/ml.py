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


def _train(df: pd.DataFrame):
    if len(df) < 20:  # per champion vaak minder data
        return None

    X = df[FEATURES].fillna(0)
    y = df["win"].astype(int)

    # als alle labels hetzelfde zijn (bv alleen wins), kan stratify falen
    if y.nunique() < 2:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    weights = pd.Series(model.coef_[0], index=FEATURES)
    weights = weights.sort_values(key=lambda s: s.abs(), ascending=False)

    return {"model": model, "auc": auc, "weights": weights}


def train_win_model(df: pd.DataFrame, champion: str | None = None):
    """
    Train model op df, optioneel gefilterd op champion.
    champion=None => alle champions
    """
    d = df.copy()

    if champion and champion != "All":
        d = d[d["champion"] == champion]

    d = d.dropna(subset=["win"])

    return _train(d)

def predict_win_proba(df: pd.DataFrame, champion: str | None = None):
    """
    Train model (optioneel champion-filter) en voorspelt win probability per game.
    Returns: (result_dict, df_with_pred)
    """
    result = train_win_model(df, champion=champion)
    if result is None:
        return None, None

    model = result["model"]

    d = df.copy()
    if champion and champion != "All":
        d = d[d["champion"] == champion].copy()

    # Safety
    d = d.dropna(subset=["win"]).copy()
    if len(d) == 0:
        return None, None

    X = d[FEATURES].fillna(0)
    d["pred_win_proba"] = model.predict_proba(X)[:, 1]
    return result, d
