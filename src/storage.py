"""Stage 4: user settings (JSON file) and study history (SQLite database)."""
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import config

SETTINGS_FILE = config.BASE_DIR / "data" / "settings.json"
DB_FILE = config.BASE_DIR / "data" / "history.db"

STYLES = ["Brief", "Normal", "Detailed"]
DEFAULTS = {"model": config.GEMINI_MODEL, "top_k": config.TOP_K, "answer_style": "Normal"}


# ------------------------------------------------------------------ settings
def load_settings(path: Path = SETTINGS_FILE) -> dict:
    """Defaults first, then anything saved in settings.json on top."""
    data = dict(DEFAULTS)
    try:
        data.update(json.loads(Path(path).read_text(encoding="utf-8")))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return validate_settings(data)


def validate_settings(s: dict) -> dict:
    """Keep every value in a safe range so a bad file can never break the app."""
    model = str(s.get("model", "")).strip() or DEFAULTS["model"]
    try:
        top_k = max(1, min(15, int(s.get("top_k", DEFAULTS["top_k"]))))
    except (TypeError, ValueError):
        top_k = DEFAULTS["top_k"]
    style = s.get("answer_style") if s.get("answer_style") in STYLES else "Normal"
    return {"model": model, "top_k": top_k, "answer_style": style}


def save_settings(s: dict, path: Path = SETTINGS_FILE) -> dict:
    clean = validate_settings(s)
    Path(path).write_text(json.dumps(clean, indent=2), encoding="utf-8")
    return clean


def apply_settings(s: dict) -> None:
    """Push settings into the running app (other modules read config at call time)."""
    config.GEMINI_MODEL = s["model"]
    config.TOP_K = int(s["top_k"])
    config.ANSWER_STYLE = s["answer_style"]


# ------------------------------------------------------------------ history
def _connect(db: Path = DB_FILE) -> sqlite3.Connection:
    con = sqlite3.connect(str(db))
    con.execute(
        """CREATE TABLE IF NOT EXISTS events (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               ts TEXT NOT NULL,
               kind TEXT NOT NULL,
               title TEXT NOT NULL,
               score INTEGER,
               total INTEGER)"""
    )
    return con


def log_event(kind: str, title: str, score: int | None = None,
              total: int | None = None, db: Path = DB_FILE) -> bool:
    """Save one activity. Never raises: history must not break the main feature."""
    try:
        with closing(_connect(db)) as con, con:
            con.execute(
                "INSERT INTO events (ts, kind, title, score, total) VALUES (?, ?, ?, ?, ?)",
                (time.strftime("%Y-%m-%d %H:%M:%S"), kind, title[:300], score, total),
            )
        return True
    except sqlite3.Error:
        return False


def get_events(limit: int = 200, db: Path = DB_FILE) -> list[dict]:
    """Newest first."""
    with closing(_connect(db)) as con:
        rows = con.execute(
            "SELECT ts, kind, title, score, total FROM events ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [{"time": r[0], "type": r[1], "what": r[2], "score": r[3], "out of": r[4]} for r in rows]


def quiz_scores(db: Path = DB_FILE) -> list[float]:
    """Quiz percentages in time order (oldest first), for the progress chart."""
    with closing(_connect(db)) as con:
        rows = con.execute(
            "SELECT score, total FROM events WHERE kind = 'quiz' AND total > 0 ORDER BY id"
        ).fetchall()
    return [round(100 * s / t, 1) for s, t in rows]


def clear_history(db: Path = DB_FILE) -> None:
    with closing(_connect(db)) as con, con:
        con.execute("DELETE FROM events")
