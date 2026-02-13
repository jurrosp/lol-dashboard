import streamlit as st
import pandas as pd
from app.config import players
from app.riot import get_puuid, get_ranked_match_ids, get_match
from app.features import extract_features
from app.analytics import build_dataframe, champion_table
from app.ml import train_win_model, predict_win_proba

st.set_page_config(page_title="LoL Dashboard", layout="wide")
st.title("League of Legends – Multi-player Dashboard (Ranked Solo/Duo)")


# ---- UI Controls ----
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    selected_player = st.selectbox("Player", players())
with col2:
    last_n = st.selectbox("Show last N ranked games", [10, 20, 30, 50], index=1)
with col3:
    refresh = st.button("Refresh (clear cache)")

if refresh:
    st.cache_data.clear()


# ---- Data Load ----
try:
    game_name, tag = selected_player.split("#", 1)
    puuid = get_puuid(game_name, tag)

    # We halen meer matches op voor ML, maar tonen alleen last_n in de UI
    want_ids = max(last_n + 25, 120)
    match_ids = get_ranked_match_ids(puuid, want=want_ids)

    records = []
    for mid in match_ids:
        match_json = get_match(mid)
        rec = extract_features(match_json, puuid)
        if rec:
            records.append(rec)

    df_all = build_dataframe(records)

except Exception as e:
    st.error(str(e))
    st.stop()

if df_all.empty:
    st.warning("Geen matches gevonden (of parsing faalde). Check Riot ID / queue / key.")
    st.stop()


# ---- UI DataFrame (laatste N voor charts/tables) ----
df = (
    df_all.sort_values("game_creation", ascending=False)
    .head(last_n)
    .sort_values("game_creation")
)

# ---- KPI Cards (op df = last_n) ----
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


# ---- Charts ----
st.subheader("Winrate Trend (Rolling 10)")
st.line_chart(df.set_index("game_creation")["winrate_roll10"])

st.subheader("Damage & Gold Per Minute")
st.line_chart(df.set_index("game_creation")[["dpm", "gpm"]])

st.subheader("Champion Pool (shown games)")
st.dataframe(champion_table(df).head(10), use_container_width=True)

st.subheader("Recent games (shown)")
cols = ["champion", "win", "kills", "deaths", "assists", "kda", "kp", "dpm", "gpm", "vision_score"]
st.dataframe(
    df.sort_values("game_creation", ascending=False)[cols].head(20),
    use_container_width=True,
)


# ---- ML Section (op df_all = zoveel mogelijk) ----
st.subheader("AI Performance Model (Champion-aware)")

st.caption(f"Training set size: {len(df_all)} games (fetched). UI is showing last {len(df)} games.")

# Champion selector voor ML
champ_list = ["All"] + sorted(df_all["champion"].dropna().unique().tolist())
selected_champ = st.selectbox("Train model for champion", champ_list, index=0)

ml_result = train_win_model(df_all, champion=selected_champ)

if ml_result is None:
    st.info("Te weinig games of te weinig win/loss variatie voor dit (champion) model.")
else:
    st.metric("Model AUC", f"{ml_result['auc']:.2f}")

    st.write("Feature impact (positief = geassocieerd met wins; negatief = met losses)")
    st.dataframe(ml_result["weights"].to_frame("weight"), use_container_width=True)

    weights = ml_result["weights"]

    st.subheader("Coaching Insights (per champion)")

    if "deaths_per_10" in weights.index and weights["deaths_per_10"] < 0:
        st.write("🔻 **Deaths/10** is negatief: minder deaths (zeker early) lijkt je winrate te helpen.")

    if "gpm" in weights.index and weights["gpm"] > 0:
        st.write("💰 **GPM** is positief: tempo/farm + goede resets lijken belangrijk voor wins.")

    if "kp" in weights.index and weights["kp"] > 0:
        st.write("🤝 **KP** is positief: meebewegen met fights/objectives loont.")

    if "vision_score" in weights.index and weights["vision_score"] > 0:
        st.write("👁 **Vision** is positief: vision blijft waardevol (zeker jungle).")

st.subheader("Predicted win probability per game")

ml_res, df_pred = predict_win_proba(df_all, champion=selected_champ)

if df_pred is None:
    st.info("Kan geen win probability plot maken (te weinig data/variatie).")
else:
    # Gebruik dezelfde last_n view voor de plot (recent)
    dfp = (
        df_pred.sort_values("game_creation", ascending=False)
        .head(last_n)
        .sort_values("game_creation")
    )

    # Line chart van predicted proba
    st.line_chart(dfp.set_index("game_creation")["pred_win_proba"])

    # Overzicht tabel (laatste N)
    view_cols = ["champion", "win", "pred_win_proba", "kills", "deaths", "assists", "kp", "dpm", "gpm"]
    st.dataframe(
        dfp.sort_values("game_creation", ascending=False)[view_cols],
        use_container_width=True,
    )

    # Quick calibration: gemiddelde predicted proba voor wins vs losses
    win_mean = df_pred[df_pred["win"] == 1]["pred_win_proba"].mean()
    loss_mean = df_pred[df_pred["win"] == 0]["pred_win_proba"].mean()
    cA, cB = st.columns(2)
    cA.metric("Avg predicted proba (wins)", f"{win_mean:.2f}" if pd.notna(win_mean) else "n/a")
    cB.metric("Avg predicted proba (losses)", f"{loss_mean:.2f}" if pd.notna(loss_mean) else "n/a")
