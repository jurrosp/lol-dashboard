import streamlit as st

# Riot routing voor EU accounts
ROUTING = "europe"

# Ranked solo/duo queue id
QUEUE_RANKED = 420

# Timeout voor API calls
TIMEOUT = 20


def riot_api_key() -> str:
    """
    Haalt Riot API key uit Streamlit secrets.
    """
    if "RIOT_API_KEY" not in st.secrets:
        raise RuntimeError("RIOT_API_KEY ontbreekt in Streamlit secrets.")
    return st.secrets["RIOT_API_KEY"].strip()


def players() -> list[str]:
    """
    Voeg hier players toe in RiotID formaat: GameName#TAG
    """
    return [
        "Evil Wim#jotul",
    ]
