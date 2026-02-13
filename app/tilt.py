import pandas as pd


def _zscore(value, mean, std):
    if std is None or std == 0 or pd.isna(std):
        return 0.0
    return (value - mean) / std


def detect_tilt(df: pd.DataFrame, recent_n: int = 7, baseline_n: int = 60):
    """
    Returns dict met tilt_score, flags, en samenvatting.
    Verwacht kolommen: win, kda, deaths_per_10, (optioneel) pred_win_proba
    """
    if df is None or df.empty or len(df) < max(15, recent_n + 5):
        return None

    d = df.sort_values("game_datetime").copy()

    # neem baseline uit het verleden (excl. de meest recente games)
    recent = d.tail(recent_n)
    baseline_pool = d.iloc[:-recent_n]
    if len(baseline_pool) < 10:
        baseline_pool = d.head(max(10, len(d) - recent_n))

    baseline = baseline_pool.tail(min(baseline_n, len(baseline_pool)))

    # metrics
    rec_win = recent["win"].mean()
    base_win = baseline["win"].mean()

    rec_kda = recent["kda"].mean()
    base_kda = baseline["kda"].mean()

    rec_d10 = recent["deaths_per_10"].mean()
    base_d10 = baseline["deaths_per_10"].mean()

    # z-scores (performance drop)
    z_kda = _zscore(rec_kda, baseline["kda"].mean(), baseline["kda"].std())
    z_d10 = _zscore(rec_d10, baseline["deaths_per_10"].mean(), baseline["deaths_per_10"].std())

    flags = []
    score = 0

    # Winrate drop
    if (base_win - rec_win) >= 0.20 and recent_n >= 5:
        flags.append("Winrate drop (recent << baseline)")
        score += 2

    # KDA drop (negatief z)
    if z_kda <= -0.8:
        flags.append("KDA significantly down vs baseline")
        score += 2

    # Deaths/10 up (positief z)
    if z_d10 >= 0.8:
        flags.append("Deaths/10 significantly up vs baseline")
        score += 2

    # Optional: predicted win probability drop
    if "pred_win_proba" in d.columns and d["pred_win_proba"].notna().any():
        rec_p = recent["pred_win_proba"].mean()
        base_p = baseline["pred_win_proba"].mean()
        z_p = _zscore(rec_p, baseline["pred_win_proba"].mean(), baseline["pred_win_proba"].std())
        if z_p <= -0.8 or (base_p - rec_p) >= 0.10:
            flags.append("Predicted win probability dropped")
            score += 1
    else:
        rec_p = base_p = None

    # Interpretatie
    if score >= 5:
        level = "HIGH"
    elif score >= 3:
        level = "MEDIUM"
    elif score >= 1:
        level = "LOW"
    else:
        level = "NONE"

    return {
        "level": level,
        "score": score,
        "recent_n": recent_n,
        "baseline_n": min(baseline_n, len(baseline)),
        "recent_winrate": rec_win,
        "baseline_winrate": base_win,
        "recent_kda": rec_kda,
        "baseline_kda": base_kda,
        "recent_deaths10": rec_d10,
        "baseline_deaths10": base_d10,
        "recent_pred": rec_p,
        "baseline_pred": base_p,
        "flags": flags,
    }