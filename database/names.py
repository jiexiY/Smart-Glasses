import math
import sqlite3
import unicodedata
from typing import Any


def name_key(name: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", name).casefold().split())


def initialize_names(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS identity_names (
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            name_key TEXT NOT NULL,
            name TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('pending', 'confirmed', 'rejected')),
            first_seen REAL NOT NULL,
            last_seen REAL NOT NULL,
            PRIMARY KEY(user_id, name_key)
        );
        CREATE TABLE IF NOT EXISTS name_observations (
            event_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            name_key TEXT NOT NULL,
            observed_at REAL NOT NULL,
            eligible INTEGER NOT NULL,
            reason TEXT NOT NULL,
            evidence TEXT NOT NULL,
            FOREIGN KEY(user_id, name_key) REFERENCES identity_names(user_id, name_key)
                ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS name_observations_candidate
            ON name_observations(user_id, name_key, observed_at);
    """)


def observe_name(
    conn: sqlite3.Connection, *, user_id: str, name: str, event_id: str,
    observed_at: float, eligible: bool, reason: str, evidence: str, mode: str,
) -> dict[str, Any]:
    key = name_key(name)
    if (not key or len(key) > 100 or not event_id or not math.isfinite(observed_at)
            or mode not in {"shadow", "auto"}):
        raise ValueError("Invalid name observation")
    # one utterance can never vote for two different identities or names
    conn.execute("BEGIN IMMEDIATE")
    existing = conn.execute(
        "SELECT user_id, name_key FROM name_observations WHERE event_id = ?", (event_id,)
    ).fetchone()
    if existing:
        return {"status": "duplicate", "user_id": existing[0], "name_key": existing[1]}
    conn.execute("""
        INSERT INTO identity_names VALUES (?, ?, ?, 'pending', ?, ?)
        ON CONFLICT(user_id, name_key) DO UPDATE SET
            first_seen = MIN(first_seen, excluded.first_seen),
            last_seen = MAX(last_seen, excluded.last_seen)
    """, (user_id, key, name.strip(), observed_at, observed_at))
    conn.execute("INSERT INTO name_observations VALUES (?, ?, ?, ?, ?, ?, ?)",
                 (event_id, user_id, key, observed_at, int(eligible), reason, evidence))
    status, last_seen = conn.execute(
        "SELECT status, last_seen FROM identity_names WHERE user_id = ? AND name_key = ?",
        (user_id, key),
    ).fetchone()
    if status != "pending":
        return {"status": status, "user_id": user_id, "name_key": key}
    # require two distinct utterance times in a bounded evidence window
    count, first, last = conn.execute("""
        SELECT COUNT(DISTINCT observed_at), MIN(observed_at), MAX(observed_at)
        FROM name_observations WHERE user_id = ? AND name_key = ? AND eligible = 1
            AND observed_at BETWEEN ? AND ?
    """, (user_id, key, last_seen - 300.0, last_seen)).fetchone()
    if eligible and count >= 2 and last - first >= 1.0:
        if mode == "auto":
            conn.execute("""UPDATE identity_names SET status = 'confirmed'
                WHERE user_id = ? AND name_key = ?""", (user_id, key))
            status = "confirmed"
        else:
            status = "would_confirm"
    return {"status": status, "user_id": user_id, "name_key": key, "reason": reason}
