import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

CLUSTER_FEATURES = [
    "kp",
    "dpm",
    "gpm",
    "vision_score",
    "deaths_per_10",
    "team_dmg_pct",
]


def cluster_playstyles(df: pd.DataFrame, k: int = 4):
    if df is None or df.empty:
        return None, None

    d = df.dropna(subset=["win"]).copy()

    if len(d) < max(20, k * 5):
        return None, None

    X = d[CLUSTER_FEATURES].fillna(0)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    d["cluster"] = km.fit_predict(Xs)

    cluster_summary = (
        d.groupby("cluster")
        .agg(
            games=("win", "size"),
            winrate=("win", "mean"),
            avg_kp=("kp", "mean"),
            avg_dpm=("dpm", "mean"),
            avg_gpm=("gpm", "mean"),
            avg_vision=("vision_score", "mean"),
            avg_deaths10=("deaths_per_10", "mean"),
            avg_teamdmg=("team_dmg_pct", "mean"),
        )
        .sort_values(["winrate", "games"], ascending=False)
        .reset_index()
    )

    def interpret_style(row):
        if row["avg_gpm"] > cluster_summary["avg_gpm"].mean() and row["avg_deaths10"] < cluster_summary["avg_deaths10"].mean():
            return "Tempo / Farm Control"
        if row["avg_kp"] > cluster_summary["avg_kp"].mean():
            return "Teamfight / High KP"
        if row["avg_vision"] > cluster_summary["avg_vision"].mean():
            return "Vision / Utility"
        if row["avg_deaths10"] > cluster_summary["avg_deaths10"].mean():
            return "High Risk / Aggressive"
        return "Balanced"

    cluster_summary["style_name"] = cluster_summary.apply(interpret_style, axis=1)

    label_map = dict(zip(cluster_summary["cluster"], cluster_summary["style_name"]))
    d["style_name"] = d["cluster"].map(label_map)

    return d, cluster_summary
