import os
import time
import sqlite3
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = (os.getenv("RIOT_API_KEY") or "").strip()
ROUTING = "europe"

GAME_NAME = "Evil Wim"  # pas aan indien nodig
TAG_LINE  = "jotul"

headers = {"X-Riot-Token": API_KEY}

TARGET_QUEUE = 420  # ranked solo/duo

def riot_get(url: str, params=None, timeout=20, max_retries=5):
    for _ in range(max_retries):
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", "1")))
            continue
        return r
    return r

def get_puuid():
    url = f"https://{ROUTING}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{GAME_NAME}/{TAG_LINE}"
    r = riot_get(url)
    if r.status_code != 200:
        raise RuntimeError(f"get_puuid failed: {r.status_code} {r.text}")
    return r.json()["puuid"]

def get_match_ids_ranked(puuid: str, total: int = 200):
    """Haalt ranked match IDs op met pagination (count max 100)."""
    url = f"https://{ROUTING}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"

    ids = []
    start = 0
    while len(ids) < total:
        batch_count = min(100, total - len(ids))
        r = riot_get(url, params={"start": start, "count": batch_count, "queue": TARGET_QUEUE})
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

def find_me(match_json: dict, puuid: str):
    for p in match_json["info"]["participants"]:
        if p["puuid"] == puuid:
            return p
    return None

def match_exists(con, match_id: str) -> bool:
    cur = con.cursor()
    cur.execute("SELECT 1 FROM matches WHERE match_id = ? LIMIT 1", (match_id,))
    return cur.fetchone() is not None

def main():
    puuid = get_puuid()
    print("PUUID:", puuid)

    con = sqlite3.connect("lol.db")
    cur = con.cursor()

    # player upsert
    cur.execute(
        "INSERT OR REPLACE INTO players(puuid, game_name, tag_line) VALUES (?, ?, ?)",
        (puuid, GAME_NAME, TAG_LINE),
    )
    con.commit()

    match_ids = get_match_ids_ranked(puuid, total=200)
    print("Fetched ranked match IDs:", len(match_ids))

    inserted = 0
    skipped = 0

    for match_id in match_ids:
        if match_exists(con, match_id):
            skipped += 1
            continue

        match_json = get_match(match_id)
        info = match_json["info"]

        # Sanity: queueId moet 420 zijn (we filteren al, maar extra zekerheid)
        if info.get("queueId") != TARGET_QUEUE:
            continue

        cur.execute(
            "INSERT INTO matches(match_id, queue_id, game_creation, game_duration) VALUES (?, ?, ?, ?)",
            (match_id, info.get("queueId"), info.get("gameCreation"), info.get("gameDuration")),
        )

        me = find_me(match_json, puuid)
        if me:
            cur.execute(
                """INSERT INTO participant_stats(
                       match_id, puuid, champion_name, win, kills, deaths, assists, lane, role
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    match_id,
                    puuid,
                    me.get("championName"),
                    1 if me.get("win") else 0,
                    me.get("kills", 0),
                    me.get("deaths", 0),
                    me.get("assists", 0),
                    me.get("lane"),
                    me.get("role"),
                ),
            )

        con.commit()
        inserted += 1

    con.close()
    print(f"Done. Inserted: {inserted}, Skipped: {skipped}")
    print("DB file: lol.db")

if __name__ == "__main__":
    main()
