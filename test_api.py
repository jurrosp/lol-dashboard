import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = (os.getenv("RIOT_API_KEY") or "").strip()
ROUTING = "europe"

GAME_NAME = "Evil Wim"
TAG_LINE = "jotul"

headers = {"X-Riot-Token": API_KEY}

def get_puuid():
    url = f"https://{ROUTING}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{GAME_NAME}/{TAG_LINE}"
    r = requests.get(url, headers=headers, timeout=20)
    if r.status_code != 200:
        print("Error get_puuid:", r.status_code, r.text)
        return None
    return r.json()["puuid"]

def get_matches(puuid, count=5):
    url = f"https://{ROUTING}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
    r = requests.get(url, headers=headers, params={"start": 0, "count": count}, timeout=20)
    if r.status_code != 200:
        print("Error get_matches:", r.status_code, r.text)
        return None
    return r.json()

def get_match_detail(match_id):
    url = f"https://{ROUTING}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    r = requests.get(url, headers=headers, timeout=20)
    if r.status_code != 200:
        print("Error get_match_detail:", r.status_code, r.text)
        return None
    return r.json()

def find_my_participant(match_json, puuid):
    participants = match_json["info"]["participants"]
    for p in participants:
        if p["puuid"] == puuid:
            return p
    return None

if __name__ == "__main__":
    puuid = get_puuid()
    print("PUUID:", puuid)

    match_ids = get_matches(puuid, count=5)
    print("Latest matches:", match_ids)

    # Pak de nieuwste match
    match_id = match_ids[0]
    match_json = get_match_detail(match_id)

    me = find_my_participant(match_json, puuid)
    if not me:
        print("Could not find your participant in match:", match_id)
        raise SystemExit(1)

    # Print jouw kernstats
    champ = me.get("championName")
    win = me.get("win")
    k = me.get("kills")
    d = me.get("deaths")
    a = me.get("assists")
    lane = me.get("lane")
    role = me.get("role")
    queue_id = match_json["info"].get("queueId")
    game_duration = match_json["info"].get("gameDuration")

    print("\n=== Latest match summary ===")
    print("Match:", match_id)
    print("QueueId:", queue_id, "| Duration(s):", game_duration)
    print("Champion:", champ)
    print("Win:", win)
    print("K/D/A:", f"{k}/{d}/{a}")
    print("Lane/Role:", lane, "/", role)