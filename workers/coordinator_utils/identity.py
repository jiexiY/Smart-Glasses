from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

import numpy as np

from core import config, config_audio, config_vision
from workers.coordinator_utils.association import closest_frame, voice_matches


MAX_FRAME_AGE = 1.0
RETRY_WINDOW = 5.0
UNKNOWN_MIN_FRAMES = 3
UNKNOWN_MIN_DURATION = 0.5


@dataclass
class PendingIdentity:
    args: dict[str, Any]
    timestamp: float
    voice_embedding: np.ndarray | None
    track_id: int
    last_seen: float
    request_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class RegisteredTrack:
    user_id: str
    first_seen: float
    last_seen: float


@dataclass
class IdentityResolution:
    status: Literal["known", "unknown", "unresolved"]
    user_id: str | None = None
    face: dict[str, Any] | None = None


def valid_embedding(embedding: Any) -> bool:
    return (
        isinstance(embedding, np.ndarray)
        and embedding.ndim == 1
        and embedding.size > 0
        and np.issubdtype(embedding.dtype, np.floating)
        and bool(np.all(np.isfinite(embedding)))
        and bool(np.isfinite(np.linalg.norm(embedding)))
        and float(np.linalg.norm(embedding)) > 0.0
    )


def resolve_identity(
    vision_cache: Sequence[tuple[float, list[dict[str, Any]]]],
    timestamp: float,
    users: dict[str, str],
    voices: dict[str, list[np.ndarray]],
    voice_embedding: np.ndarray | None,
    is_self_intro: bool,
    registered_tracks: dict[int, RegisteredTrack],
) -> IdentityResolution:
    frame = closest_frame(vision_cache, timestamp)
    faces = frame[1] if frame and abs(frame[0] - timestamp) <= MAX_FRAME_AGE else []
    face = faces[0] if len(faces) == 1 else None

    # the wearer's greeting cannot identify the person being greeted
    voice_ids: set[str] = set()
    if is_self_intro and valid_embedding(voice_embedding):
        for uid, embeddings in voices.items():
            valid = [emb for emb in embeddings if valid_embedding(emb)
                     and emb.shape == voice_embedding.shape]
            if uid in users and voice_matches(
                voice_embedding, valid, is_self_intro=True,
                threshold=config_audio.SIMILARITY_THRESHOLD,
            ):
                voice_ids.add(uid)
    if len(voice_ids) > 1:
        return IdentityResolution("unresolved", face=face)

    face_id = None
    if face is not None:
        uid = face.get("user_id", config.DEFAULT_ID)
        registered = registered_tracks.get(face.get("track_id"))
        if uid != config.DEFAULT_ID:
            score = face.get("score", 0.0)
            if (uid not in users or not np.isfinite(score)
                    or score <= config_vision.CONFIDENCE_THRESHOLD_MATCHING):
                return IdentityResolution("unresolved", face=face)
            if registered and registered.user_id != uid:
                return IdentityResolution("unresolved", face=face)
            face_id = uid
        elif (registered and registered.first_seen - MAX_FRAME_AGE
              <= timestamp <= registered.last_seen + MAX_FRAME_AGE):
            if registered.user_id not in users:
                return IdentityResolution("unresolved", face=face)
            face_id = registered.user_id

    voice_id = next(iter(voice_ids), None)
    if face_id and voice_id and face_id != voice_id:
        return IdentityResolution("unresolved", face=face)
    user_id = face_id or voice_id
    if user_id:
        return IdentityResolution("known", user_id, face)
    if face is not None and isinstance(face.get("track_id"), int) and valid_embedding(face.get("emb")):
        return IdentityResolution("unknown", face=face)
    return IdentityResolution("unresolved", face=face)


def stable_unknown_track(
    vision_cache: Sequence[tuple[float, list[dict[str, Any]]]],
    track_id: int,
    timestamp: float,
) -> bool:
    observations: list[float] = []
    for frame_time, faces in vision_cache:
        if not timestamp - RETRY_WINDOW <= frame_time <= timestamp:
            continue
        if (len(faces) != 1 or faces[0].get("track_id") != track_id
                or faces[0].get("user_id", config.DEFAULT_ID) != config.DEFAULT_ID
                or not valid_embedding(faces[0].get("emb"))):
            observations.clear()
            continue
        if observations and frame_time - observations[-1] > MAX_FRAME_AGE:
            observations.clear()
        if not observations or frame_time > observations[-1]:
            observations.append(frame_time)
    return (len(observations) >= UNKNOWN_MIN_FRAMES
            and observations[-1] - observations[0] >= UNKNOWN_MIN_DURATION)
