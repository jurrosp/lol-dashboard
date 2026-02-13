import sqlite3
from fastapi import FastAPI, Query

DB_PATH = "lol.db"
TARGET_QUEUE_DEFAULT = 420

app = FastAPI(title="LoL Dashboard API")

def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

@app.get("/players")
def list_players():
    con = db()
    cur = con.cursor()
    cur.execute("SELECT puuid, game_name, tag_line FROM players ORDER BY game_name")
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows

@app.get("/stats/summary")
def summary(
    puuid: str = Query(...),
    queue_id: int = Query(TARGET_QUEUE_DEFAULT),
    limit: int = Query(20, ge=1, le=100),
):
    con = db()
    cur = con.cursor()
    cur.execute("""
        SELECT ps.win, ps.kills, ps.deaths, ps.assists
        FROM participant_stats ps
        JOIN matches m ON m.match_id = ps.match_id
        WHERE ps.puuid = ? AND m.queue_id = ?
        ORDER BY m.game_creation DESC
        LIMIT ?
    """, (puuid, queue_id, limit))
    rows = cur.fetchall()
    con.close()

    games = len(rows)
    wins = sum(r["win"] for r in rows)
    kills = sum(r["kills"] for r in rows)
    deaths = sum(r["deaths"] for r in rows)
    assists = sum(r["assists"] for r in rows)

    return {
        "puuid": puuid,
        "queue_id": queue_id,
        "limit": limit,
        "games": games,
        "wins": wins,
        "winrate": (wins / games) if games else 0,
        "avg_kills": (kills / games) if games else 0,
        "avg_deaths": (deaths / games) if games else 0,
        "avg_assists": (assists / games) if games else 0,
    }

@app.get("/stats/champions")
def champions(
    puuid: str = Query(...),
    queue_id: int = Query(TARGET_QUEUE_DEFAULT),
    limit: int = Query(20, ge=1, le=100),
):
    con = db()
    cur = con.cursor()
    cur.execute("""
        SELECT ps.champion_name AS champion,
               COUNT(*) AS games,
               SUM(ps.win) AS wins
        FROM participant_stats ps
        JOIN matches m ON m.match_id = ps.match_id
        WHERE ps.puuid = ? AND m.queue_id = ?
        GROUP BY ps.champion_name
        ORDER BY games DESC
        LIMIT ?
    """, (puuid, queue_id, limit))
    rows = [dict(r) for r in cur.fetchall()]
    con.close()

    # voeg winrate per champ toe
    for r in rows:
        r["winrate"] = (r["wins"] / r["games"]) if r["games"] else 0

    return rows
