def find_participant(match_json: dict, puuid: str):
    for p in match_json["info"]["participants"]:
        if p["puuid"] == puuid:
            return p
    return None


def extract_features(match_json: dict, puuid: str) -> dict | None:
    """
    Extract per-match features voor jouw dashboard (jungle/performance oriented).
    """
    me = find_participant(match_json, puuid)
    if not me:
        return None

    info = match_json["info"]
    ch = me.get("challenges", {}) or {}

    record = {
        "match_id": match_json["metadata"]["matchId"],
        "game_creation": info.get("gameCreation"),
        "duration_s": info.get("gameDuration"),
        "champion": me.get("championName"),
        "win": int(bool(me.get("win"))),
        "kills": int(me.get("kills", 0)),
        "deaths": int(me.get("deaths", 0)),
        "assists": int(me.get("assists", 0)),
        "kp": float(ch.get("killParticipation", 0.0)),
        "dpm": float(ch.get("damagePerMinute", 0.0)),
        "gpm": float(ch.get("goldPerMinute", 0.0)),
        "team_dmg_pct": float(ch.get("teamDamagePercentage", 0.0)),
        "vision_score": float(me.get("visionScore", 0.0)),
        "jungle_cs": int(me.get("neutralMinionsKilled", 0)),
    }

    record["kda"] = (record["kills"] + record["assists"]) / max(record["deaths"], 1)
    record["deaths_per_10"] = (record["deaths"] / max(record["duration_s"], 1)) * 600

    return record
