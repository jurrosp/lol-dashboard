import sqlite3

con = sqlite3.connect("lol.db")
cur = con.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS players (
  puuid TEXT PRIMARY KEY,
  game_name TEXT NOT NULL,
  tag_line TEXT NOT NULL
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS matches (
  match_id TEXT PRIMARY KEY,
  queue_id INTEGER,
  game_creation INTEGER,
  game_duration INTEGER
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS participant_stats (
  match_id TEXT NOT NULL,
  puuid TEXT NOT NULL,
  champion_name TEXT,
  win INTEGER,
  kills INTEGER,
  deaths INTEGER,
  assists INTEGER,
  lane TEXT,
  role TEXT,
  PRIMARY KEY (match_id, puuid),
  FOREIGN KEY (match_id) REFERENCES matches(match_id),
  FOREIGN KEY (puuid) REFERENCES players(puuid)
);
""")

con.commit()
con.close()
print("DB initialized: lol.db")
