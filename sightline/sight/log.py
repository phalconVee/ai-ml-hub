"""An append-only decision log: one JSON object per line (JSONL).

Append-only means a human correction never overwrites the model's original
answer: it is a NEW row pointing at the same frame and question. So the log
keeps both what Jev said and what a person decided, which is exactly the data
you'd later use to tune the thresholds.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from sight import config
from sight.decisions import Decision


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_decisions(frame_id: str, model: str, decisions: list[Decision],
                  path: Path | None = None) -> list[dict]:
    """Append one row per decision; return the rows (the UI shows them too)."""
    rows = [{
        "time": now(), "frame": frame_id, "model": model,
        "question": d.question_id, "type": d.kind, "subject": d.subject,
        "answer": d.answer,
        "confidence": None if d.value is None else round(d.value, 3),
        "status": d.status, "reason": d.reason,
    } for d in decisions]
    _append(path, rows)
    return rows


def log_review(frame_id: str, question_id: str, subject: str, model_answer: str,
               human_answer: str, path: Path | None = None) -> dict:
    """Record a person's verdict on a queued decision."""
    row = {
        "time": now(), "frame": frame_id, "model": "human",
        "question": question_id, "type": "review", "subject": subject,
        "answer": human_answer, "confidence": None,
        "status": "confirmed" if human_answer == model_answer else "corrected",
        "reason": f"Jev said {model_answer!r}",
    }
    _append(path, [row])
    return row


def read_log(path: Path | None = None, limit: int = 500) -> list[dict]:
    """Newest first, at most `limit` rows."""
    path = path or config.LOG_PATH   # resolved at call time, so tests can redirect it
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in reversed(lines[-limit:]) if line.strip()]


def _append(path: Path | None, rows: list[dict]) -> None:
    path = path or config.LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
