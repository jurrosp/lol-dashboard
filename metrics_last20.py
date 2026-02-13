import os
import time
import requests
from collections import Counter, defaultdict
from dotenv import load_dotenv

load_dotenv()

API_KEY = (os.getenv("RIOT_API_KEY") or "").strip()
ROUTING = "europe"

GAME_NAME = "Evil Wim"
TAG_LINE = "jotul"

headers = {"X-Riot-Token": API_KEY}

def riot_get(url: str, params=None, timeout=20, max_retries=5):
    """GET met simpele 429 handling."""
    for _ in range(max_retries):
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", "1"))
            time.sleep(retry_after)
            continue
        return r
    return r


def get_match_ids(puuid: str, total: int = 200):
    url = f"https://{ROUTING}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"

    ids = []
    start = 0
    while len(ids) < total:
        batch_count = min(100, total - len(ids))

        r = riot_get(url, params={
            "start": start,
            "count": batch_count,
            "queue": 420   # <- deze regel toegevoegd
        })

        if r.status_code != 200:
            raise RuntimeError(f"get_match_ids failed: {r.status_code} {r.text}")

        batch = r.json()
        if not batch:
            break

        ids.extend(batch)
        start += batch_count

    return ids

def get_match(match_id: str):
    url = f"https://{ROUTING}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    r = riot_get(url)
    if r.status_code != 200:
        raise RuntimeError(f"get_match failed for {match_id}: {r.status_code} {r.text}")
    return r.json()

def find_me(match_json: dict, puuid: str) -> dict | None:
    for p in match_json["info"]["participants"]:
        if p["puuid"] == puuid:
            return p
    return None

def main():
    puuid = "7Kd05L3B0dy-WexXoA8IDgBDzttNgGJEIh8oIQ3r0gNGMuhWDJshDGyRLPE5-M3ZCm2R02MlWBMoAA"
    print("PUUID:", puuid)

    match_ids = get_match_ids(puuid, total=200)  # pak wat extra, we filteren ranked
    print("Fetched match IDs:", len(match_ids))

    # Metrics accumulators
    games = 0
    wins = 0
    kills = deaths = assists = 0
    champ_games = Counter()
    champ_wins = Counter()

    # We willen specifiek ranked solo/duo: queueId 420
    TARGET_QUEUE = 420

    for match_id in match_ids:
        match_json = get_match(match_id)

        queue_id = match_json["info"].get("queueId")
        if queue_id != TARGET_QUEUE:
            continue

        me = find_me(match_json, puuid)
        if not me:
            continue

        games += 1
        win = bool(me.get("win"))
        wins += 1 if win else 0

        k = int(me.get("kills", 0))
        d = int(me.get("deaths", 0))
        a = int(me.get("assists", 0))

        kills += k
        deaths += d
        assists += a

        champ = me.get("championName", "Unknown")
        champ_games[champ] += 1
        if win:
            champ_wins[champ] += 1

        # stop als we 20 ranked games hebben
        if games >= 20:
            break

    if games == 0:
        print("Geen ranked solo (queue 420) matches gevonden in deze batch.")
        return

    winrate = wins / games
    avg_k = kills / games
    avg_d = deaths / games
    avg_a = assists / games

    print("\n=== Dashboard metrics (last ranked games) ===")
    print("Games:", games)
    print("Wins:", wins)
    print(f"Winrate: {winrate*100:.1f}%")
    print(f"Avg K/D/A: {avg_k:.2f}/{avg_d:.2f}/{avg_a:.2f}")

    print("\nTop champions (games | winrate):")
    for champ, g in champ_games.most_common(5):
        w = champ_wins[champ]
        wr = (w / g) if g else 0
        print(f"- {champ}: {g} | {wr*100:.0f}%")

if __name__ == "__main__":
    main()
