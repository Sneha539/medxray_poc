import sqlite3
from pathlib import Path

DB_PATH = Path("predictions.db")

def _connect():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT,
            probs TEXT,
            image_name TEXT,
            overlay_name TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()
    conn.close()

def log_prediction(label: str, probs_json: str, image_name: str, overlay_name: str | None):
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO predictions (label, probs, image_name, overlay_name) VALUES (?, ?, ?, ?)",
        (label, probs_json, image_name, overlay_name),
    )
    conn.commit()
    conn.close()

def get_history(limit: int = 50):
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, label, probs, image_name, overlay_name, timestamp "
        "FROM predictions ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows
