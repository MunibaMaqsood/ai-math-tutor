import json

from src.storage import (clear_history, get_events, load_settings, log_event,
                         quiz_scores, save_settings, validate_settings)


def test_settings_round_trip(tmp_path):
    f = tmp_path / "s.json"
    save_settings({"model": " my-model ", "top_k": 7, "answer_style": "Brief"}, f)
    s = load_settings(f)
    assert s == {"model": "my-model", "top_k": 7, "answer_style": "Brief"}


def test_bad_settings_are_repaired(tmp_path):
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"model": "", "top_k": 999, "answer_style": "Weird"}))
    s = load_settings(f)
    assert s["top_k"] == 15 and s["answer_style"] == "Normal" and s["model"]
    assert validate_settings({"top_k": "abc"})["top_k"] >= 1
    f.write_text("not json")
    assert load_settings(f)["answer_style"] == "Normal"


def test_history(tmp_path):
    db = tmp_path / "h.db"
    assert log_event("question", "What is gcd?", db=db)
    assert log_event("quiz", "Euclid (Medium)", 3, 5, db=db)
    assert log_event("quiz", "Bayes (Hard)", 4, 5, db=db)
    events = get_events(db=db)
    assert events[0]["what"] == "Bayes (Hard)" and len(events) == 3  # newest first
    assert quiz_scores(db=db) == [60.0, 80.0]  # oldest first
    clear_history(db=db)
    assert get_events(db=db) == []
