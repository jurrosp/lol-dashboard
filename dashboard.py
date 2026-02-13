import streamlit as st
import pandas as pd

from app.config import players
from app.riot import get_puuid, get_ranked_match_ids, get_match
from app.features import extract_features
from app.analytics import build_dataframe, champion_table
from app.ml import train_win_model, predict_win_proba
from app.clustering import cluster_playstyles
from app.tilt import detect_tilt


# -----------------------------
# Page + Sidebar
# -----------------------------
st.set_page_config(page_title="LoL Dashboard", layout="wide")
st.title("League of Legends Dashboard")

with st.sidebar:
    st.header("Settings")
    selected_player = st.selectbox("Player", players())
    last_n = st.selectbox("Show last N games", [10, 20, 30, 50], index=1)

    refresh = st.button("Refresh (clear cache)")
    if refresh:
        st.cache_data.clear()

    st.divider()
    st.caption("Voeg spelers toe in app/config.py → players().")


# -----------------------------
# Data Fetch + Build df_all/df
# -----------------------------
try:
    game_name, tag = selected_player.split("#", 1)
    puuid = get_puuid(game_name, tag)

    # We halen meer IDs op voor ML/Clustering/Tilt (maar tonen last_n in UI)
    want_ids = max(last_n + 25, 120)
    match_ids = get_ranked_match_ids(puuid, want=want_ids)

    records = []
    failed = 0

    for mid in match_ids:
        try:
            match_json = get_match(mid)
        except Exception:
            failed += 1
            continue

        rec = extract_features(match_json, puuid)
        if rec:
            records.append(rec)

    df_all = build_dataframe(records)

    if failed > 0:
        st.warning(f"Riot API transient errors. Skipped {failed} matches while fetching.")

except Exception as e:
    st.error(str(e))
    st.stop()

if df_all is None or df_all.empty:
    st.warning("Geen matches gevonden (of parsing faalde).")
    st.stop()

# UI slice: laatste N voor grafieken/tables in Overview
df = (
    df_all.sort_values("game_datetime", ascending=False)
    .head(last_n)
    .sort_values("game_datetime")
)

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_ml, tab_playstyles, tab_tilt = st.tabs(
    ["Overview", "ML", "Playstyles", "Tilt"]
)


# -----------------------------
# Tab: Overview
# -----------------------------
with tab_overview:
    st.subheader("Overview (last N shown games)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Games (shown)", int(len(df)))
    c2.metric("Winrate", f"{df['win'].mean()*100:.1f}%")
    c3.metric("Avg KP%", f"{df['kp'].mean()*100:.1f}%")
    c4.metric("Avg KDA", f"{df['kda'].mean():.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg DPM", f"{df['dpm'].mean():.0f}")
    c6.metric("Avg GPM", f"{df['gpm'].mean():.0f}")
    c7.metric("Avg Vision", f"{df['vision_score'].mean():.1f}")
    c8.metric("Deaths/10", f"{df['deaths_per_10'].mean():.2f}")

    st.subheader("Winrate Trend (Rolling 10)")
    st.line_chart(df.set_index("game_datetime")["winrate_roll10"])

    st.subheader("Damage & Gold Per Minute")
    st.line_chart(df.set_index("game_datetime")[["dpm", "gpm"]])

    st.subheader("Champion Pool (shown)")
    st.dataframe(champion_table(df).head(12), use_container_width=True)

    st.subheader("Recent games (shown)")
    cols = ["game_datetime", "champion", "win", "kills", "deaths", "assists", "kda", "kp", "dpm", "gpm", "vision_score"]
    st.dataframe(
        df.sort_values("game_datetime", ascending=False)[cols].head(25),
        use_container_width=True,
    )


# -----------------------------
# Tab: ML
# -----------------------------
with tab_ml:
    st.subheader("ML (Champion-aware)")

    st.caption(f"Fetched games (df_all): {len(df_all)} | Shown (df): {len(df)}")

    champ_list = ["All"] + sorted(df_all["champion"].dropna().unique().tolist())
    selected_champ = st.selectbox("Train / view model for champion", champ_list, index=0)

    ml_result = train_win_model(df_all, champion=selected_champ)

    if ml_result is None:
        st.info("Te weinig games of te weinig win/loss variatie voor dit (champion) model.")
    else:
        st.metric("Model AUC", f"{ml_result['auc']:.2f}")
        st.write("Feature impact (positief = geassocieerd met wins; negatief = met losses)")
        st.dataframe(ml_result["weights"].to_frame("weight"), use_container_width=True)

        weights = ml_result["weights"]
        st.subheader("Coaching Insights")
        if "deaths_per_10" in weights.index and weights["deaths_per_10"] < 0:
            st.write("🔻 Minder deaths (zeker early) correleert met wins.")
        if "gpm" in weights.index and weights["gpm"] > 0:
            st.write("💰 Tempo/farm (GPM) correleert positief met wins.")
        if "kp" in weights.index and weights["kp"] > 0:
            st.write("🤝 Kill Participation correleert positief met wins.")
        if "vision_score" in weights.index and weights["vision_score"] > 0:
            st.write("👁 Vision correleert positief met wins.")

    st.subheader("Predicted win probability per game")
    _, df_pred = predict_win_proba(df_all, champion=selected_champ)

    if df_pred is None:
        st.info("Kan geen win probability plot maken (te weinig data/variatie).")
    else:
        dfp = (
            df_pred.sort_values("game_datetime", ascending=False)
            .head(last_n)
            .sort_values("game_datetime")
        )

        st.line_chart(dfp.set_index("game_datetime")["pred_win_proba"])

        view_cols = ["game_datetime", "champion", "win", "pred_win_proba", "kp", "dpm", "gpm", "deaths_per_10"]
        st.dataframe(
            dfp.sort_values("game_datetime", ascending=False)[view_cols],
            use_container_width=True,
        )

        win_mean = df_pred[df_pred["win"] == 1]["pred_win_proba"].mean()
        loss_mean = df_pred[df_pred["win"] == 0]["pred_win_proba"].mean()
        cA, cB = st.columns(2)
        cA.metric("Avg predicted proba (wins)", f"{win_mean:.2f}" if pd.notna(win_mean) else "n/a")
        cB.metric("Avg predicted proba (losses)", f"{loss_mean:.2f}" if pd.notna(loss_mean) else "n/a")


# -----------------------------
# Prepare df_all_with_pred for Tilt (using All-champ model)
# We do this OUTSIDE tabs so Tilt can always use it.
# -----------------------------
df_all_with_pred = df_all.copy()
_, df_pred_all = predict_win_proba(df_all_with_pred, champion="All")
if df_pred_all is not None and "pred_win_proba" in df_pred_all.columns:
    df_all_with_pred = df_all_with_pred.merge(
        df_pred_all[["match_id", "pred_win_proba"]],
        on="match_id",
        how="left",
        suffixes=("", "_new"),
    )
    if "pred_win_proba_new" in df_all_with_pred.columns:
        df_all_with_pred["pred_win_proba"] = df_all_with_pred["pred_win_proba_new"].combine_first(
            df_all_with_pred.get("pred_win_proba")
        )
        df_all_with_pred = df_all_with_pred.drop(columns=["pred_win_proba_new"])


# -----------------------------
# Tab: Playstyles
# -----------------------------
with tab_playstyles:
    st.subheader("Playstyles (KMeans clustering)")

    k = st.selectbox("Number of playstyles (K)", [3, 4, 5], index=1)

    df_clustered, cluster_summary = cluster_playstyles(df_all, k=k)

    if df_clustered is None:
        st.info("Te weinig games voor clustering (richtlijn: minimaal ~20-25).")
    else:
        st.subheader("Playstyle Legend (cluster → style)")
        legend = cluster_summary[["cluster", "style_name", "games", "winrate"]].copy()
        legend["winrate_pct"] = (legend["winrate"] * 100).round(1)

        st.dataframe(
            legend[["cluster", "style_name", "games", "winrate_pct"]].rename(columns={"winrate_pct": "winrate (%)"}),
            use_container_width=True,
        )

        best = legend.sort_values(["winrate", "games"], ascending=False).iloc[0]
        st.success(
            f"Beste playstyle: **{best['style_name']}** (cluster {int(best['cluster'])}) — "
            f"{int(best['games'])} games — {best['winrate_pct']:.1f}% WR"
        )

        st.subheader("Winrate per playstyle")
        winrate_by_style = (
            legend.groupby("style_name")
            .agg(games=("games", "sum"), winrate=("winrate", "mean"))
            .reset_index()
            .sort_values("winrate", ascending=False)
        )
        winrate_by_style["winrate_pct"] = winrate_by_style["winrate"] * 100
        st.bar_chart(winrate_by_style.set_index("style_name")["winrate_pct"])

        st.subheader("Recent games: detected playstyle")
        recent = (
            df_clustered.sort_values("game_datetime", ascending=False)
            .head(last_n)
            .sort_values("game_datetime")
        )
        cols = ["game_datetime", "champion", "win", "style_name", "kp", "dpm", "gpm", "vision_score", "deaths_per_10"]
        st.dataframe(
            recent.sort_values("game_datetime", ascending=False)[cols],
            use_container_width=True,
        )

        st.caption("style_name is een interpretatie van clusters; zie legend voor context.")


# -----------------------------
# Tab: Tilt
# -----------------------------
with tab_tilt:
    st.subheader("Tilt / Performance Instability")

    recent_n = st.selectbox("Recent window (games)", [5, 7, 10], index=1)
    baseline_n = st.selectbox("Baseline window (games)", [30, 60, 80], index=1)

    tilt = detect_tilt(df_all_with_pred, recent_n=recent_n, baseline_n=baseline_n)

    if tilt is None:
        st.info("Te weinig data voor tilt analyse.")
    else:
        level = tilt["level"]
        score = tilt["score"]

        if level == "HIGH":
            st.error(f"⚠️ Tilt risk: {level} (score {score}) — overweeg pauze / review.")
        elif level == "MEDIUM":
            st.warning(f"⚠️ Tilt risk: {level} (score {score}) — speel bewust, evt. 1 game max.")
        elif level == "LOW":
            st.info(f"Tilt risk: {level} (score {score}) — lichte afwijking t.o.v. baseline.")
        else:
            st.success(f"Tilt risk: {level} (score {score}) — stabiele performance.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Winrate recent", f"{tilt['recent_winrate']*100:.1f}%")
        c1.metric("Winrate baseline", f"{tilt['baseline_winrate']*100:.1f}%")

        c2.metric("KDA recent", f"{tilt['recent_kda']:.2f}")
        c2.metric("KDA baseline", f"{tilt['baseline_kda']:.2f}")

        c3.metric("Deaths/10 recent", f"{tilt['recent_deaths10']:.2f}")
        c3.metric("Deaths/10 baseline", f"{tilt['baseline_deaths10']:.2f}")

        if tilt.get("recent_pred") is not None and tilt.get("baseline_pred") is not None:
            c4, c5 = st.columns(2)
            c4.metric("Pred proba recent", f"{tilt['recent_pred']:.2f}")
            c5.metric("Pred proba baseline", f"{tilt['baseline_pred']:.2f}")

        if tilt["flags"]:
            st.write("Triggers:")
            for f in tilt["flags"]:
                st.write(f"- {f}")

st.caption("Data cached (15 min). Refresh in sidebar clears cache.")