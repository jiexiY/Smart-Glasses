import argparse
import gc
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from database.database import DatabaseManager
from test_worker_helpers import blank_state, identity_face, observe, registration_event
from workers.coordinator_utils.handlers import handle_event, retry_pending_identities


class IdentityScenarios(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.db = DatabaseManager(str(Path(self.temporary.name) / 'identities.db'))
        _, _, self.vision, self.audio, self.state = blank_state()
        self.state.db = self.db
        self.db.create_user('saved-person', 'Shrish')
        self.db.save_voice_embedding('saved-person', np.array([1.0, 0.0]))
        self.db.save_chat_history('saved-person', 'Earlier conversation')

    def tearDown(self) -> None:
        gc.collect()
        self.temporary.cleanup()

    def unknown(self, name: str = 'Shri', voice: np.ndarray | None = None) -> None:
        face = identity_face()
        for timestamp in (4.4, 4.7, 5.0):
            observe(self.state, timestamp, [face])
        handle_event(registration_event(name, 5.0, voice, self_intro=True), self.state)

    def confirm(self, timestamp: float = 5.0, request_id: str | None = None) -> None:
        if request_id is None:
            request_id = self.state.pending_identities[4].request_id
        with patch('workers.coordinator_utils.handlers.time.time', return_value=timestamp):
            handle_event({'type': 'confirm_enrollment', 'request_id': request_id,
                          'confirmed': True}, self.state)

    def assert_saved_only(self) -> None:
        self.assertEqual(self.db.get_all_users(), {'saved-person': 'Shrish'},
                         'Uncertain recognition must not create another UUID')
        self.assertEqual(self.db.get_chat_history('saved-person')[0][0], 'Earlier conversation')

    def test_1_known_face_new_nickname(self) -> None:
        observe(self.state, 5.0, [identity_face('saved-person')])
        handle_event(registration_event('Shri', self_intro=True), self.state)
        self.assert_saved_only()
        self.assertFalse(self.state.pending_identities)
        self.assertEqual(self.vision.items, [])

    def test_2_unclear_face_known_voice(self) -> None:
        self.unknown(voice=np.array([1.0, 0.0]))
        self.assert_saved_only()
        self.assertFalse(self.state.pending_identities)
        self.assertEqual(self.vision.items[0]['user_id'], 'saved-person')

    def test_3_both_unclear_wait_instead_of_duplicate(self) -> None:
        self.unknown(voice=np.array([0.0, 1.0]))
        self.assert_saved_only()
        self.assertIn(4, self.state.pending_identities)
        for timestamp in (5.3, 5.6, 5.9):
            observe(self.state, timestamp, [identity_face()])
        self.assert_saved_only()
        self.assertEqual(self.vision.items, [])
        self.assertEqual(self.audio.items, [])

    def test_4_different_person_same_name_confirmed_separately(self) -> None:
        self.unknown('Shrish', np.array([0.0, 1.0]))
        self.assert_saved_only()
        self.confirm()
        users = self.db.get_all_users()
        self.assertEqual(len(users), 2)
        self.assertEqual(set(users.values()), {'Shrish'})
        new_id = next(uid for uid in users if uid != 'saved-person')
        self.assertEqual(self.vision.items[0]['user_id'], new_id)
        self.assertEqual(self.audio.items[0]['user_id'], new_id)
        observe(self.state, 5.3, [identity_face(new_id)])
        handle_event(registration_event('Shrish', 5.3), self.state)
        self.assertEqual(self.db.get_all_users(), users)

    def test_recovery_reuses_id_and_cancels_pending(self) -> None:
        self.unknown()
        self.assert_saved_only()
        observe(self.state, 5.3, [identity_face('saved-person')])
        self.assert_saved_only()
        self.assertFalse(self.state.pending_identities)

    def test_continuous_unknown_expires_without_enrollment(self) -> None:
        self.unknown()
        for tick in range(1, 12):
            observe(self.state, 5.0 + tick * 0.5, [identity_face()])
        self.assert_saved_only()
        self.assertFalse(self.state.pending_identities)
        self.assertEqual(self.vision.items, [])
        self.assertEqual(self.audio.items, [])

    def test_confirmed_new_name_enrolls_once(self) -> None:
        self.unknown('A new person')
        self.assert_saved_only()
        request_id = self.state.pending_identities[4].request_id
        self.confirm(request_id=request_id)
        self.confirm(request_id=request_id)
        self.assertEqual(len(self.db.get_all_users()), 2)
        self.assertEqual(len(self.vision.items), 1)
        self.assertFalse(self.state.pending_identities)

    def test_llm_cannot_confirm_new_enrollment(self) -> None:
        self.unknown()
        event = registration_event('Shri')
        event['args']['confirmed'] = True
        handle_event(event, self.state)
        handle_event({'type': 'intent', 'cmd': 'CONFIRM_ENROLLMENT',
                      'args': {'confirmed': True}}, self.state)
        self.assert_saved_only()
        self.assertEqual(self.vision.items, [])

    def test_expired_confirmation_cannot_create(self) -> None:
        self.unknown('Shrish')
        request_id = self.state.pending_identities[4].request_id
        for tick in range(1, 12):
            observe(self.state, 5.0 + tick * 0.5, [identity_face()])
        self.confirm(10.5, request_id)
        self.assert_saved_only()

    def test_changed_or_stale_target_cannot_confirm(self) -> None:
        for faces, now in (([identity_face(track_id=9)], 5.2),
                           ([identity_face(), identity_face(track_id=9)], 5.2),
                           ([identity_face()], 6.2)):
            with self.subTest(faces=faces, now=now):
                self.state.pending_identities.clear()
                self.state.vision_cache.clear()
                self.unknown('Shrish')
                if now < 6:
                    observe(self.state, now, faces)
                self.confirm(now)
                self.assert_saved_only()
        self.assertEqual(self.vision.items, [])

    def test_final_check_reuses_recovered_identity(self) -> None:
        self.unknown('Shrish')
        request_id = self.state.pending_identities[4].request_id
        self.state.vision_cache.append((5.1, [identity_face('saved-person')]))
        self.confirm(5.1, request_id)
        self.assert_saved_only()
        self.assertFalse(self.state.pending_identities)

    def test_conflicting_face_voice_cannot_confirm(self) -> None:
        self.db.create_user('other-person', 'Bob')
        observe(self.state, 5.0, [identity_face('other-person')])
        handle_event(registration_event('Shri', voice=np.array([1.0, 0.0]), self_intro=True), self.state)
        self.confirm()
        self.assertEqual(self.db.get_all_users(), {'saved-person': 'Shrish', 'other-person': 'Bob'})
        self.assertEqual(self.vision.items, [])
        self.assertEqual(self.audio.items, [])

    def test_early_confirmation_is_not_saved_for_later(self) -> None:
        observe(self.state, 5.0, [identity_face()])
        handle_event(registration_event('Another person'), self.state)
        request_id = self.state.pending_identities[4].request_id
        self.confirm()
        observe(self.state, 5.3, [identity_face()])
        observe(self.state, 5.6, [identity_face()])
        self.assert_saved_only()
        self.confirm(5.6, request_id)
        self.assertEqual(len(self.db.get_all_users()), 2)

    def test_old_confirmation_cannot_apply_to_reused_track(self) -> None:
        self.unknown('Shrish')
        old_request = self.state.pending_identities[4].request_id
        retry_pending_identities(self.state, 6.1, [])
        self.assertFalse(self.state.pending_identities)
        for timestamp in (7.0, 7.3, 7.6):
            observe(self.state, timestamp, [identity_face()])
        handle_event(registration_event('Different person', 7.6), self.state)
        self.assertNotEqual(self.state.pending_identities[4].request_id, old_request)
        self.confirm(7.6, old_request)
        self.assert_saved_only()

    def test_new_voice_match_on_retry_cancels_pending(self) -> None:
        self.unknown()
        observe(self.state, 5.3, [identity_face()])
        handle_event(registration_event('Shri', 5.3, np.array([1.0, 0.0]), self_intro=True), self.state)
        self.assert_saved_only()
        self.assertFalse(self.state.pending_identities)
        self.assertEqual(self.vision.items[0]['user_id'], 'saved-person')

    def test_confirmation_requires_explicit_true(self) -> None:
        self.unknown('Shrish')
        request_id = self.state.pending_identities[4].request_id
        for value in (False, None, 1, 'true'):
            with patch('workers.coordinator_utils.handlers.time.time', return_value=5.0):
                handle_event({'type': 'confirm_enrollment', 'request_id': request_id,
                              'confirmed': value}, self.state)
        self.assert_saved_only()
        self.assertEqual(self.vision.items, [])


class ReportResult(unittest.TextTestResult):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.passed: list[str] = []

    def addSuccess(self, test: unittest.TestCase) -> None:
        super().addSuccess(test)
        self.passed.append(test.id())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(IdentityScenarios)
    result = unittest.TextTestRunner(verbosity=2, resultclass=ReportResult).run(suite)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            'scope': 'Simulated recognition/intent events; real isolated SQLite; no recognition models or AI service.',
            'tests_run': result.testsRun, 'passed': result.passed,
            'failures': [{'test': test.id(), 'trace': trace} for test, trace in result.failures],
            'errors': [{'test': test.id(), 'trace': trace} for test, trace in result.errors],
            'successful': result.wasSuccessful(),
        }, indent=2) + '\n', encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
