import streamlit as st
import random
import time
import requests
from .config import ROUTING, TIMEOUT, riot_api_key, QUEUE_RANKED


def riot_get(url: str, params=None, max_retries=7):
    """
    GET request met:
    - 429 rate-limit handling
    - 5xx (502/503/504) transient error retry met exponential backoff
    """
    headers = {"X-Riot-Token": riot_api_key()}
    backoff = 0.8  # start in seconds

    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
        except requests.RequestException:
            # network glitch: retry
            time.sleep(backoff + random.random() * 0.2)
            backoff = min(backoff * 1.8, 8)
            continue

        # Rate limit
        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", "1"))
            time.sleep(retry_after)
            continue

        # Transient server/proxy errors
        if r.status_code in (502, 503, 504):
            time.sleep(backoff + random.random() * 0.2)
            backoff = min(backoff * 1.8, 8)
            continue

        return r

    return r



@st.cache_data(ttl=900)
def get_puuid(game_name: str, tag_line: str) -> str:
    url = f"https://{ROUTING}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
    r = riot_get(url)

    if r.status_code != 200:
        raise RuntimeError(f"get_puuid failed: {r.status_code} {r.text}")

    return r.json()["puuid"]

@st.cache_data(ttl=900)
def get_ranked_match_ids(puuid: str, want: int = 60) -> list[str]:
    """
    Haalt ranked solo/duo match IDs op met pagination.
    """
    url = f"https://{ROUTING}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"

    ids = []
    start = 0

    while len(ids) < want:
        batch_size = min(100, want - len(ids))

        r = riot_get(
            url,
            params={
                "start": start,
                "count": batch_size,
                "queue": QUEUE_RANKED,
            },
        )

        if r.status_code != 200:
            raise RuntimeError(f"get_match_ids failed: {r.status_code} {r.text}")

        batch = r.json()
        if not batch:
            break

        ids.extend(batch)
        start += batch_size

    return ids

@st.cache_data(ttl=900)
def get_match(match_id: str) -> dict:
    url = f"https://{ROUTING}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    r = riot_get(url)

    if r.status_code != 200:
        raise RuntimeError(f"get_match failed: {r.status_code} {r.text}")

    return r.json()
