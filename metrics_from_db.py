import sqlite3
from collections import Counter

PUUID = "7Kd05L3B0dy-WexXoA8IDgBDzttNgGJEIh8oIQ3r0gNGMuhWDJshDGyRLPE5-M3ZCm2R02MlWBMoAA"
TARGET_QUEUE = 420

con = sqlite3.connect("lol.db")
cur = con.cursor()

cur.execute("""
SELECT ps.win, ps.kills, ps.deaths, ps.assists, ps.champion_name
FROM participant_stats ps
JOIN matches m ON m.match_id = ps.match_id
WHERE ps.puuid = ? AND m.queue_id = ?
ORDER BY m.game_creation DESC
LIMIT 20
""", (PUUID, TARGET_QUEUE))

rows = cur.fetchall()
con.close()

games = len(rows)
wins = sum(r[0] for r in rows)
kills = sum(r[1] for r in rows)
deaths = sum(r[2] for r in rows)
assists = sum(r[3] for r in rows)

champs = [r[4] for r in rows]
champ_games = Counter(champs)
champ_wins = Counter([c for (w,_,_,_,c) in rows if w == 1])

print("\n=== DB metrics (last ranked games) ===")
print("Games:", games)
print("Wins:", wins)
print(f"Winrate: {(wins/games*100):.1f}%" if games else "Winrate: n/a")
print(f"Avg K/D/A: {kills/games:.2f}/{deaths/games:.2f}/{assists/games:.2f}" if games else "Avg K/D/A: n/a")

print("\nTop champions (games | winrate):")
for champ, g in champ_games.most_common(5):
    w = champ_wins[champ]
    print(f"- {champ}: {g} | {(w/g*100):.0f}%")
