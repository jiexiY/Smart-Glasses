import json
import math
import re
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from core import config, config_vision
from database.names import name_key
from workers.coordinator_utils.association import closest_frame, cosine_sim
from workers.coordinator_utils.identity import MAX_FRAME_AGE, valid_embedding


# provisional thresholds for evaluation, not calibrated accuracy claims
NAME_VOICE_THRESHOLD = 0.70
NAME_VOICE_MARGIN = 0.15
MAX_NAME_EVENT_AGE = 60.0
NAME_POLICY = "self-name-v2"


class NameDirectory(Protocol):
    def get_all_users(self) -> dict[str, str]: ...
    def record_name_observation(self, **observation: Any) -> dict[str, Any]: ...


def capture_name_source(
    event: dict[str, Any], users: dict[str, str], voices: dict[str, list[np.ndarray]],
    vision_cache: Sequence[tuple[float, list[dict[str, Any]]]],
) -> dict[str, Any]:
    embedding = event.get("embedding")
    scores: list[tuple[float, str]] = []
    if valid_embedding(embedding):
        for uid, stored in voices.items():
            samples = [sample for sample in stored if valid_embedding(sample)
                       and sample.shape == embedding.shape]
            if uid in users and samples:
                mean = np.mean(samples, axis=0)
                if valid_embedding(mean):
                    score = cosine_sim(embedding, mean)
                    if math.isfinite(score):
                        scores.append((score, uid))
    scores.sort(reverse=True)
    best = scores[0][0] if scores else -1.0
    margin = best - (scores[1][0] if len(scores) > 1 else -1.0)
    voice_id = (scores[0][1] if scores and best >= NAME_VOICE_THRESHOLD
                and margin >= NAME_VOICE_MARGIN else None)
    timestamp = event.get("time_start")
    frame = (closest_frame(vision_cache, timestamp)
             if isinstance(timestamp, (float, int)) and math.isfinite(timestamp) else None)
    if frame is not None and not abs(frame[0] - timestamp) <= MAX_FRAME_AGE:
        frame = None
    faces = frame[1] if frame else []
    face_ids = sorted({face["user_id"] for face in faces
                       if face.get("user_id") in users})
    face = faces[0] if len(faces) == 1 else {}
    score = face.get("score")
    face_score = (float(score) if isinstance(score, (int, float, np.floating))
                  and not isinstance(score, bool) and math.isfinite(score) else None)
    face_id = (face.get("user_id") if face.get("user_id") in users
               and face_score is not None
               and face_score > config_vision.CONFIDENCE_THRESHOLD_MATCHING else None)
    return {
        "event_id": event.get("id", ""), "text": event.get("text", ""),
        "timestamp": timestamp, "voice_user_id": voice_id,
        "voice_score": best, "voice_margin": margin, "face_user_ids": face_ids,
        "face_user_id": face_id, "face_count": len(faces), "face_score": face_score,
        "face_timestamp": frame[0] if frame else None,
    }


def explicit_self_name(text: str, name: str) -> bool:
    normalized = name_key(text).replace("’", "'")
    target = name_key(name).replace("’", "'")
    if not target or target.startswith(("not ", "no ")):
        return False
    # a whole utterance prevents quoted, negated and third-person name mentions
    prefix = r"(?:(?:hi|hello|hey)[,! ]+)?(?:my name is|my nickname is|i'm|i am|call me|you can call me|i go by) "
    return re.fullmatch(prefix + re.escape(target) + r"[.!]?", normalized) is not None


def learn_name(
    directory: NameDirectory, args: dict[str, Any], source: dict[str, Any],
    now: float, mode: str,
) -> dict[str, Any]:
    if mode not in {"shadow", "auto"}:
        return {"status": "ignored", "reason": "invalid_mode"}
    name = args.get("name")
    event_id = source.get("event_id")
    timestamp = source.get("timestamp")
    text = source.get("text")
    if (not isinstance(name, str) or not name_key(name) or len(name_key(name)) > 100
            or name_key(name) == name_key(config.DEFAULT_NAME)
            or not isinstance(event_id, str) or not event_id
            or not isinstance(text, str) or not isinstance(timestamp, (int, float))
            or not math.isfinite(timestamp) or not 0 <= now - timestamp <= MAX_NAME_EVENT_AGE):
        return {"status": "ignored", "reason": "missing_or_stale_source"}
    users = directory.get_all_users()
    voice_id = source.get("voice_user_id")
    if voice_id not in users:
        voice_id = None
    face_ids = {uid for uid in source.get("face_user_ids", []) if uid in users}
    user_id = voice_id or (next(iter(face_ids)) if len(face_ids) == 1 else None)
    if user_id is None:
        return {"status": "unresolved", "reason": "no_verified_identity"}
    # a conservative first version; greetings and introductions by others stay pending
    reason = "clear_self_introduction"
    eligible = True
    if args.get("is_self_introduction") is not True or not explicit_self_name(text, name):
        reason, eligible = "name_referent_unverified", False
    elif voice_id is None:
        reason, eligible = "speaker_unverified", False
    elif source.get("face_count", 0) > 1:
        reason, eligible = "multiple_visible_faces", False
    elif source.get("face_user_id") is None:
        reason, eligible = "face_unverified", False
    elif source["face_user_id"] != voice_id:
        reason, eligible = "face_voice_disagree", False
    return directory.record_name_observation(
        user_id=user_id, name=name, event_id=event_id, observed_at=timestamp,
        eligible=eligible, reason=reason,
        evidence=json.dumps({"policy": NAME_POLICY, "source": source}, sort_keys=True), mode=mode,
    )
