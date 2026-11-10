import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S-%fZ')
    output = repo / '.test-results/identity' / stamp
    output.mkdir(parents=True)
    env = dict(os.environ, PYTHONPATH=str(repo), PYTHONIOENCODING='utf-8')
    print('SMART GLASSES - IDENTITY SITUATIONS', flush=True)
    print('Simulated recognition events; isolated test databases; no live AI calls.\n', flush=True)
    suites = [('Identity situations', 'tests/test_identity_scenarios.py',
               ['--report', str(output / 'scenarios.json')]),
              ('Worker helpers and alias replay', 'tests/test_worker_helpers.py', []),
              ('Database', 'tests/test_database.py', []),
              ('API queue with fake clients', 'tests/test_api_worker_queue.py', [])]
    completed = []
    for index, (label, script, args) in enumerate(suites, start=1):
        print(f'{index}. {label} ...', flush=True)
        log = output / f'{Path(script).stem}.log'
        with log.open('w', encoding='utf-8') as handle:
            result = subprocess.run([sys.executable, '-u', str(repo / script), *args],
                                    cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            print(log.read_text(encoding='utf-8', errors='replace')[-10000:], flush=True)
            raise SystemExit(f'FAIL: {label}. Log: {log}')
        completed.append(label)
        print('   PASS', flush=True)
    scenarios = json.loads((output / 'scenarios.json').read_text(encoding='utf-8'))
    summary = '\n'.join([
        'Identity test results', '', f'{scenarios["tests_run"]} identity tests passed. All four suites passed.', '',
        '1. Same face, new nickname: existing ID kept.',
        '2. Unclear face, recognized voice: existing ID kept.',
        '3. Both unclear: waits, retries, then expires without a new ID.',
        '4. Different person, same name: separate ID only after explicit enrollment.', '',
        'Recovery, expiry, stale/duplicate confirmations and conflicting evidence are covered.',
        'New-person enrollment requires the local confirmation event. No glasses confirmation UI is included.',
        'These are logic tests, not a recognition-accuracy or end-to-end AI test.',
    ])
    (output / 'summary.md').write_text(summary + '\n', encoding='utf-8')
    (repo / '.test-results/identity/latest-run.txt').write_text(str(output) + '\n', encoding='utf-8')
    print('\n' + summary, flush=True)
    print(f'\nReport: {output / "summary.md"}', flush=True)


if __name__ == '__main__':
    main()
