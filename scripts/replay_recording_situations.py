import argparse
import gc
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
import tempfile
import types
import wave
from collections import Counter, deque
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from core import config, config_audio, config_vision
from database.database import DatabaseManager
from scripts.replay_name_learning import Commands
from workers.audio_utils.speaker import cosine_sim
from workers.audio_utils.utterance import pcm_from_bytes
from workers.audio_utils.vad import voice_embedding
from workers.coordinator_utils import handlers
from workers.coordinator_utils.name_learning import capture_name_source
from workers.vision_utils.inspireface_processor import InspireFaceProcessor
from workers.vision_utils.pipeline import decode_frame, recognize_frame

BEFORE = '2e44f9f032730ab0b13fa0a8a1de8c671d2a5f87'
CLIPS = {
    'riley': {'stem': 'riley_self_intro', 'reference_at': 5.0, 'speech': (3.36, 4.96)},
    'shrish': {'stem': 'shrish_introduced', 'reference_at': 0.109, 'speech': (3.2, 4.32)},
}
CASES = [
    {'id': '1_known_face_new_name', 'clip': 'shrish', 'video': 'clean', 'voice': 'none', 'kind': 'known_face'},
    {'id': '2_blurred_face_known_voice', 'clip': 'riley', 'video': 'blur31', 'voice': 'same_utterance', 'kind': 'known_voice'},
    {'id': '3_blurred_face_weak_voice', 'clip': 'riley', 'video': 'blur31', 'voice': 'separate_utterance', 'kind': 'both_unclear'},
    {'id': '3b_pixelated_face_weak_voice', 'clip': 'riley', 'video': 'pixel80', 'voice': 'separate_utterance', 'kind': 'both_unclear'},
    {'id': '3c_blur_then_clear', 'clip': 'riley', 'video': 'recover6', 'voice': 'separate_utterance', 'kind': 'recovery'},
    {'id': '4_different_face_same_name', 'clip': 'riley', 'video': 'clean', 'voice': 'none', 'kind': 'same_name'},
    {'id': 'noise_10db', 'clip': 'riley', 'video': 'blur31', 'voice': 'separate_utterance', 'snr': 10, 'kind': 'noise'},
    {'id': 'noise_minus10db', 'clip': 'riley', 'video': 'blur31', 'voice': 'separate_utterance', 'snr': -10, 'kind': 'noise'},
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def processor_for(db: DatabaseManager, model: Path) -> InspireFaceProcessor:
    with patch('workers.vision_utils.inspireface_processor.DatabaseManager', lambda: db):
        return InspireFaceProcessor(model_path=str(model))


def transform(frame: np.ndarray, mode: str, timestamp: float) -> np.ndarray:
    if mode == 'blur31' or (mode == 'recover6' and timestamp < 6.0):
        return cv2.GaussianBlur(frame, (31, 31), 0)
    if mode == 'pixel80':
        return cv2.resize(cv2.resize(frame, (80, 60)), (640, 480))
    return frame


def read_audio(path: Path) -> np.ndarray:
    with wave.open(str(path), 'rb') as file:
        assert (file.getframerate(), file.getnchannels(), file.getsampwidth()) == (16000, 1, 2)
        return pcm_from_bytes(file.readframes(file.getnframes()))


def write_audio(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), 'wb') as file:
        file.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        file.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())


def noisy(samples: np.ndarray, snr: int) -> np.ndarray:
    noise = np.random.default_rng(8).standard_normal(len(samples)).astype(np.float32)
    noise *= np.sqrt(np.mean(samples ** 2)) / (10 ** (snr / 20) * np.sqrt(np.mean(noise ** 2)))
    result = samples + noise
    return result / max(1.0, float(np.max(np.abs(result))))


def seed_db(path: Path, case: dict[str, Any], reference: np.ndarray,
            voice: np.ndarray | None) -> DatabaseManager:
    db = DatabaseManager(str(path))
    name = ('Riley' if case['kind'] == 'same_name' else
            'Shrish' if case['clip'] == 'shrish' else 'Riley Stored')
    db.create_user('saved-person', name)
    db.save_face_embedding('saved-person', reference)
    if voice is not None:
        db.save_voice_embedding('saved-person', voice)
    db.save_chat_history('saved-person', 'Existing history must remain attached')
    return db


def replay(module: Any, case: dict[str, Any], frames: list[dict[str, Any]],
           reference: np.ndarray, saved_voice: np.ndarray | None, query_voice: np.ndarray,
           text: str, candidate: str | None, db_path: Path) -> dict[str, Any]:
    db = seed_db(db_path, case, reference, saved_voice)
    state = module.CoordinatorState(
        db, Commands(), Commands(), Commands(), deque(), deque(), deque(maxlen=10), None,
        10.0, 30.0, 20.0, 640*480, 320, 240, 400, 0,
    )
    start, end = CLIPS[case['clip']]['speech']
    received = end + 0.2
    sent, confirmation_attempted = False, False
    source, pending_at_intent, pending_peak, recovered = {}, 0, 0, False
    initial_name = db.get_user_name('saved-person')
    with redirect_stdout(StringIO()):
        for observation in frames:
            at = observation['at']
            with patch.object(module.time, 'time', return_value=at):
                module.handle_event({'type': 'vision_result', 'faces': observation['faces']}, state)
                if not sent and at >= received:
                    source = capture_name_source(
                        {'id': case['id'], 'time_start': start, 'text': text, 'embedding': query_voice},
                        db.get_all_users(), db.get_all_voices(), state.vision_cache,
                    )
                    if candidate:
                        module.handle_event({
                            'type': 'intent', 'cmd': 'REGISTER_IDENTITY', 'timestamp': start,
                            'voice_embedding': query_voice, 'name_source': source,
                            'args': {'name': candidate, 'is_self_introduction': case['clip'] == 'riley',
                                     'speaker_name': config.DEFAULT_NAME},
                        }, state)
                    sent = True
                    pending_at_intent = len(state.pending_identities)
                if (case['kind'] == 'same_name' and sent and not confirmation_attempted
                        and at >= received + 0.6 and len(state.pending_identities) == 1):
                    pending = next(iter(state.pending_identities.values()))
                    module.handle_event({'type': 'confirm_enrollment', 'confirmed': True,
                                         'request_id': getattr(pending, 'request_id', 'unsupported-before-fix')}, state)
                    confirmation_attempted = True
                pending_peak = max(pending_peak, len(state.pending_identities))
                if (case['kind'] == 'recovery' and 6 <= at <= start + 5
                        and any(f['user_id'] == 'saved-person' for f in observation['faces'])
                        and pending_at_intent and not state.pending_identities
                        and len(db.get_all_users()) == 1):
                    recovered = True
    users = db.get_all_users()
    voice_score = None if saved_voice is None else cosine_sim(query_voice, saved_voice)
    return {
        'users_before': 1, 'users_after': len(users), 'new_user_count': len(users)-1,
        'saved_name_unchanged': users.get('saved-person') == initial_name,
        'saved_history_unchanged': db.get_chat_history('saved-person')[0][0] == 'Existing history must remain attached',
        'pending_at_intent': pending_at_intent, 'pending_peak': pending_peak,
        'pending_at_end': len(state.pending_identities), 'recovered_before_expiry': recovered,
        'confirmation_injected': confirmation_attempted, 'intent_injected': bool(candidate),
        'name_source': source, 'voice_similarity_to_reference': voice_score,
        'separate_same_name_ids': len(users) == 2 and len(set(users.values())) == 1,
        'face_commands': len(state.vision_commands.items),
    }


def check_case(case: dict[str, Any], current: dict[str, Any]) -> tuple[bool, bool]:
    source = current['name_source']
    preserved = current['saved_name_unchanged'] and current['saved_history_unchanged']
    no_duplicate = current['users_after'] == 1 and preserved
    kind = case['kind']
    if kind != 'noise' and not current['intent_injected']:
        return False, False
    if kind == 'known_face':
        exercised = source.get('face_user_id') == 'saved-person'
        return exercised, exercised and no_duplicate
    if kind == 'known_voice':
        exercised = source.get('face_user_id') is None and source.get('voice_user_id') == 'saved-person'
        return exercised, exercised and no_duplicate
    if kind in ('both_unclear', 'recovery'):
        exercised = (current['pending_at_intent'] > 0 and source.get('face_user_id') is None
                     and current['voice_similarity_to_reference'] <= config_audio.SIMILARITY_THRESHOLD)
        passed = exercised and no_duplicate and current['pending_at_end'] == 0
        return exercised, passed and (kind != 'recovery' or current['recovered_before_expiry'])
    if kind == 'same_name':
        exercised = current['pending_at_intent'] > 0 and current['confirmation_injected']
        return exercised, exercised and current['separate_same_name_ids'] and preserved
    return bool(current['intent_injected']), no_duplicate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--face-model', type=Path, default=REPO.parents[1] / 'work/inspireface/models/Megatron')
    parser.add_argument('--asr-model', type=Path, default=REPO.parents[1] / 'work/parakeet-tdt-v3-int8')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    assert args.face_model.is_file() and args.asr_model.is_dir(), 'Provide --face-model and --asr-model paths to local models'
    output = (args.output or REPO / '.test-results/recording-situations' /
              datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S-%fZ')).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output/'media').mkdir(exist_ok=True)
    report = {
        'scope': 'Real video/audio models on controlled recording variants, followed by isolated coordinator replay.',
        'limits': ['Registration intent and new-person confirmation are supplied by the harness; UF AI is not called.',
                   'Reference faces come from the same recordings. No held-out identity accuracy claim.',
                   'The known-voice control reuses the enrollment utterance; the weak-voice condition uses another utterance assigned by the fixture, not verified speaker labels.',
                   'Same-name collision is a database fixture label; it is not a claim about participants names.',
                   'Windows INT8 ONNX Parakeet is an alternative to the production MLX backend.',
                   'Utterance windows come from the earlier VAD run; VAD is not rerun on noisy audio.',
                   'Audio and video use file-zero time origins; intent arrival is fixed at utterance end plus 0.2 seconds, not measured live latency.',
                   'Model outputs are replayed without live worker queues; gallery feedback after enrollment is not evaluated.'],
        'repo_commit': subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
        'before_commit': BEFORE, 'script_sha256': sha256(Path(__file__)),
        'packages': {p: importlib.metadata.version(p) for p in ('inspireface','onnxruntime','onnx-asr','numpy')},
        'model_sha256': {'face': sha256(args.face_model),
                         'voice': sha256(REPO/config_audio.REDIMNET_PATH),
                         'voice_data': sha256(REPO/(config_audio.REDIMNET_PATH+'.data')),
                         'asr_encoder': sha256(args.asr_model/'encoder-model.int8.onnx'),
                         'asr_decoder': sha256(args.asr_model/'decoder_joint-model.int8.onnx'),
                         'asr_vocabulary': sha256(args.asr_model/'vocab.txt'),
                         'asr_config': sha256(args.asr_model/'config.json'),
                         'asr_preprocessor': sha256(args.asr_model/'nemo128.onnx')},
        'thresholds': {'face': config_vision.CONFIDENCE_THRESHOLD_MATCHING, 'voice': config_audio.SIMILARITY_THRESHOLD},
        'sources': {}, 'cases': [],
    }
    def save() -> None:
        (output/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    before = types.ModuleType('recording_situations_before')
    sys.modules[before.__name__] = before
    source = subprocess.check_output(['git','-C',str(REPO),'show',f'{BEFORE}:workers/coordinator_utils/handlers.py'],text=True,encoding='utf-8')
    exec(compile(source, 'before-handler.py', 'exec'), before.__dict__)
    before_identity = types.ModuleType('recording_situations_before_identity')
    sys.modules[before_identity.__name__] = before_identity
    identity_source = subprocess.check_output(['git','-C',str(REPO),'show',
                                               f'{BEFORE}:workers/coordinator_utils/identity.py'],text=True,encoding='utf-8')
    exec(compile(identity_source,'before-identity.py','exec'),before_identity.__dict__)
    for name in ('MAX_FRAME_AGE','RETRY_WINDOW','PendingIdentity','RegisteredTrack',
                 'resolve_identity','stable_unknown_track','valid_embedding'):
        setattr(before,name,getattr(before_identity,name))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads, opts.inter_op_num_threads = 4, 1
    voice_model = ort.InferenceSession(str(REPO/config_audio.REDIMNET_PATH),opts,providers=['CPUExecutionProvider'])
    import onnx_asr
    asr = onnx_asr.load_model('nemo-parakeet-tdt-0.6b-v3', args.asr_model,
                              quantization='int8',providers=['CPUExecutionProvider'],sess_options=opts)
    audio, references = {}, {}
    print(f'REAL RECORDING SITUATIONS\nResults: {output}',flush=True)
    with tempfile.TemporaryDirectory(prefix='sg-recording-situations-') as folder:
        temporary = Path(folder)
        for label, details in CLIPS.items():
            video = REPO/'api/simulator_resources'/f'{details["stem"]}_92.mp4'
            wav = REPO/'api/simulator_resources'/f'{details["stem"]}.wav'
            audio[label] = read_audio(wav)
            report['sources'][label] = {'video_sha256':sha256(video),'audio_sha256':sha256(wav),
                                        'full_transcript':asr.recognize(audio[label],sample_rate=16000),
                                        'speech_range':details['speech'],'reference_at':details['reference_at']}
            db = DatabaseManager(str(temporary/f'ref-{label}.db'))
            processor = processor_for(db,args.face_model)
            capture = cv2.VideoCapture(str(video))
            fps = capture.get(cv2.CAP_PROP_FPS)
            index = 0
            while True:
                ok, frame = capture.read()
                assert ok, f'No reference face found for {label}'
                if index/fps >= details['reference_at']:
                    faces = processor.detect_faces(frame)
                    if len(faces) == 1:
                        references[label] = processor.extract_embedding(frame,faces[0])
                        cv2.imwrite(str(output/'media'/f'{label}-reference.jpg'),frame)
                        report['sources'][label]['reference_frame'] = index
                        break
                index += 1
            capture.release()
            processor.session.release()
        weak_samples = audio['riley'][12*16000:round(15.84*16000)]
        weak_voice = voice_embedding(voice_model,weak_samples[None,:])
        write_audio(output/'media'/'weak-voice-reference.wav',weak_samples)
        for case in CASES:
            print(f'Running {case["id"]} ...',flush=True)
            details = CLIPS[case['clip']]
            start,end = details['speech']
            clean = audio[case['clip']][round(start*16000):round(end*16000)]
            query = noisy(clean,case['snr']) if 'snr' in case else clean
            write_audio(output/'media'/f'{case["id"]}.wav',query)
            text = asr.recognize(query,sample_rate=16000)
            match = re.fullmatch(r"Hey,? I'm (.+?)\.",text)
            candidate = match.group(1) if match else ('Shri' if text == 'Hey Shri.' else None)
            query_voice = voice_embedding(voice_model,query[None,:])
            saved_voice = (voice_embedding(voice_model,clean[None,:]) if case['voice']=='same_utterance'
                           else weak_voice if case['voice']=='separate_utterance' else None)
            reference = references['shrish' if case['kind']=='same_name' else case['clip']]
            db = seed_db(temporary/f'{case["id"]}-model.db',case,reference,saved_voice)
            processor = processor_for(db,args.face_model)
            processor.session.set_track_lost_recovery_mode(True)
            capture = cv2.VideoCapture(str(REPO/'api/simulator_resources'/f'{details["stem"]}_92.mp4'))
            fps = capture.get(cv2.CAP_PROP_FPS)
            writer = cv2.VideoWriter(str(output/'media'/f'{case["id"]}.mp4'),cv2.VideoWriter.fourcc(*'mp4v'),fps,(640,480))
            assert writer.isOpened()
            active, frames, counts = {}, [], Counter()
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                at = index/fps
                frame = transform(cv2.resize(frame,(640,480)),case['video'],at)
                writer.write(frame)
                encoded, packet = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,70])
                assert encoded
                with patch('workers.vision_utils.pipeline.time.time',return_value=at):
                    faces = recognize_frame(processor,active,decode_frame(packet.tobytes()),
                                            recheck_interval=2.,lost_after=1.,confidence_threshold=.5)
                frames.append({'at':at,'faces':faces})
                counts['frames'] += 1
                counts['no_face_frames'] += not faces
                counts['known_face_frames'] += any(f['user_id']=='saved-person' for f in faces)
                counts['unknown_face_frames'] += any(f['user_id']==config.DEFAULT_ID for f in faces)
                index += 1
            capture.release()
            writer.release()
            processor.session.release()
            row = {**case,'transcript':text,'candidate_name':candidate,'vision':dict(counts),
                   'media_video':f'media/{case["id"]}.mp4','media_audio':f'media/{case["id"]}.wav'}
            for version,module in [('before',before),('current',handlers)]:
                row[version] = replay(module,case,frames,reference,saved_voice,query_voice,text,candidate,
                                      temporary/f'{case["id"]}-{version}.db')
            row['condition_exercised'],row['passed'] = check_case(case,row['current'])
            trace = [{'at':round(f['at'],4),'faces':[{
                'track_id':int(face['track_id']),'user_id':face['user_id'],'score':float(face['score']),
                'fresh_embedding':face['emb'] is not None} for face in f['faces']]} for f in frames]
            (output/f'{case["id"]}-faces.json').write_text(json.dumps(trace,indent=2)+'\n',encoding='utf-8')
            report['cases'].append(row)
            save()
            print(f"  {'PASS' if row['passed'] else 'NOT VERIFIED'}; condition={row['condition_exercised']}; users before fix={row['before']['users_after']}, now={row['current']['users_after']}; transcript={text!r}",flush=True)
            gc.collect()
        gc.collect()
    core_ids = {'1_known_face_new_name','2_blurred_face_known_voice','3_blurred_face_weak_voice','4_different_face_same_name'}
    report['core_four_passed'] = all(r['passed'] for r in report['cases'] if r['id'] in core_ids)
    report['all_safety_checks_passed'] = all(r['passed'] for r in report['cases'])
    save()
    lines = ['Recording situations: actual model results','',
             '| Situation | Condition observed | Result | Users before fix / now |',
             '| --- | --- | --- | --- |']
    for r in report['cases']:
        lines.append(f"| {r['id']} | {r['condition_exercised']} | {'PASS' if r['passed'] else 'NOT VERIFIED'} | {r['before']['users_after']} / {r['current']['users_after']} |")
    lines += ['', 'These checks used actual video/audio inference with fixed transformations and unchanged thresholds.',
              'The instruction to register a transcribed name was supplied by the harness, not a live AI service.',
              'The known-voice case replays the enrollment utterance and is a positive control, not independent voice accuracy.',
              'The weak-voice reference is a different segment assigned to that fixture; speaker ownership is not human-verified.',
              'A noise case with no name produced no registration instruction; it does not exercise the duplicate branch.',
              'The same-name situation assigns the other recording reference the same display name and injects confirmation.',
              'MP4 files contain the video variant only. WAV files contain the tested utterance; its original time range is in results.json.',
              'Raw scores, exact transcripts, conditions, file hashes and face timelines are saved beside this report.']
    (output/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (REPO/'.test-results/recording-situations').mkdir(parents=True,exist_ok=True)
    (REPO/'.test-results/recording-situations/latest-run.txt').write_text(str(output)+'\n',encoding='utf-8')
    print('\n'+'\n'.join(lines),flush=True)
    print(f'\nReport: {output/"summary.md"}',flush=True)
    raise SystemExit(0 if report['all_safety_checks_passed'] else 1)


if __name__ == '__main__':
    main()
