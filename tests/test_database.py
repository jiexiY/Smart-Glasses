import gc
import os
import tempfile
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from database.database import DatabaseManager


def test_identity_names() -> None:
    with tempfile.TemporaryDirectory() as folder:
        path = str(Path(folder) / "migration.db")
        # upgrade an existing users table without replacing its records
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE users (id TEXT PRIMARY KEY, name TEXT NOT NULL)")
            conn.execute("INSERT INTO users VALUES ('p1', 'Shrish Shah')")
        conn.close()
        db = DatabaseManager(path)
        db.create_user("p2", "Another Person")
        original = db.get_all_users()

        def observe(event_id: str, at: float, **changes: object) -> dict:
            data = dict(user_id="p1", name="Shri", event_id=event_id, observed_at=at,
                        eligible=True, reason="test_evidence", evidence="Call me Shri", mode="auto")
            data.update(changes)
            return db.record_name_observation(**data)

        assert observe("one", 10)["status"] == "pending"
        assert db.get_user_ids_by_name("Shri") == []
        assert observe("one", 10, user_id="p2")["status"] == "duplicate"
        assert db.get_identity_names("p2") == []
        assert observe("two", 12)["status"] == "confirmed"
        assert db.get_user_ids_by_name(" SHRI ") == ["p1"]
        assert db.find_user_candidates("shah") == ["p1"]
        assert db.get_all_users() == original
        db.reject_identity_name("p1", "Shri")
        assert db.get_user_ids_by_name("Shri") == []
        assert observe("three", 14)["status"] == "rejected"
        assert db.get_identity_names("p1")[0]["status"] == "rejected"

        with ThreadPoolExecutor(max_workers=2) as executor:
            statuses = list(executor.map(lambda _: observe("race", 20, name="Sam")["status"], range(2)))
        assert sorted(statuses) == ["duplicate", "pending"]
        assert observe("next", 22, name="Sam")["status"] == "confirmed"
        assert observe("other-one", 24, name="Sam", user_id="p2")["status"] == "pending"
        assert observe("other-two", 26, name="Sam", user_id="p2")["status"] == "confirmed"
        assert db.get_user_ids_by_name("sam") == ["p1", "p2"]
        reloaded = DatabaseManager(path)
        assert reloaded.get_user_ids_by_name("sam") == ["p1", "p2"]
        db.delete_user("p1")
        assert db.get_user_ids_by_name("sam") == ["p2"]
        with sqlite3.connect(path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM name_observations WHERE user_id = 'p1'").fetchone()[0] == 0
        conn.close()
        gc.collect()


def run_tests() -> None:
    test_identity_names()
    print("--- Starting Database Tests ---")

    db_dir = tempfile.mkdtemp()
    db_path = os.path.join(db_dir, "identities.db")
    db = DatabaseManager(db_path=db_path)
    db.create_user("Alice", "Alice")
    print("Database initialized successfully")

    user_name = "Alice"

    dummy_face_emb_1 = np.random.rand(128).astype(np.float32)
    dummy_face_emb_2 = np.random.rand(128).astype(np.float32)
    dummy_voice_emb = np.random.rand(256).astype(np.float32)

    print("Saving embeddings")
    db.save_face_embedding(user_name, dummy_face_emb_1)
    db.save_face_embedding(user_name, dummy_face_emb_2)
    db.save_voice_embedding(user_name, dummy_voice_emb)

    all_faces = db.get_all_faces()
    all_voices = db.get_all_voices()

    assert user_name in all_faces, "User not found in face dictionary"
    assert len(all_faces[user_name]) == 2, "Should have exactly 2 face embeddings"

    assert np.array_equal(all_faces[user_name][0], dummy_face_emb_1), (
        "Face embedding 1 corrupted!"
    )
    assert np.array_equal(all_faces[user_name][1], dummy_face_emb_2), (
        "Face embedding 2 corrupted!"
    )
    assert np.array_equal(all_voices[user_name][0], dummy_voice_emb), (
        "Voice embedding corrupted!"
    )

    print("NumPy Array Adapter/Converter working; array conversion successful")

    print("Saving chat history...")
    db.save_chat_history(user_name, "Hello, system!")
    db.save_chat_history(user_name, "What is my name?")

    chat_history = db.get_chat_history(user_name, limit=5)

    assert len(chat_history) == 2, "Should have exactly 2 chat entries"
    assert chat_history[0][0] == "What is my name?", "Chat order or text mismatch!"
    assert chat_history[1][0] == "Hello, system!", "Chat text mismatch!"

    print("Chat history saved and retrieved in correct order (newest first).")

    # release sqlite handles before removing temporary files on Windows
    gc.collect()
    for suffix in ("", "-shm", "-wal"):
        path = db_path + suffix
        if os.path.exists(path):
            os.remove(path)
    os.rmdir(db_dir)
    print("Test database cleaned up")
    print("--- 🎉 All Tests Passed! 🎉 ---")


if __name__ == "__main__":
    run_tests()
