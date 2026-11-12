"""Bounded same-scope dispatcher supervision; never retries scientific workers."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/hy-tmp/resource_allocation_results/chapter2/gpu0_five_seed_150k_20261104/runtime/python/bin/python'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def queue_locked(root):
    """Only inspect the lock; dispatcher remains its sole runtime owner."""
    with (root / 'queue.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock, fcntl.LOCK_UN)
        return False


def blocker(root):
    states = read(root / 'dispatch_state.json', {})
    blocked = {job: state.get('phase') for job, state in states.items()
               if state.get('phase') in ('failed', 'needs_identity_review', 'spawning')}
    if blocked:
        return dict(reason='scientific failure or unresolved worker identity; no automatic relaunch', jobs=blocked)
    return None


def complete(root):
    plan = read(root / 'plan.json', {})
    states = read(root / 'dispatch_state.json', {})
    jobs = plan.get('jobs', [])
    return bool(jobs) and all(states.get(job['id'], {}).get('phase') == 'complete' for job in jobs)


def failure_blocker(root, since):
    failure = read(root / 'controller_failure.json', {})
    if failure.get('time', 0) < since:
        return None
    traceback = failure.get('traceback', '')
    guarded = ('ValueError:', 'AssertionError:', 'identity mismatch', 'identity audit',
               'Frozen source mismatch', 'receipt changed', 'refusing a duplicate',
               'full-spend', 'full spend', 'native replay')
    if any(fragment in traceback for fragment in guarded):
        return dict(reason='dispatcher reported a scientific, provenance or identity error', failure=failure)
    return None


def supervise(root, command, *, max_restarts=2, poll_seconds=3.):
    """The CLI fixes root/command; arguments here permit isolated engineering tests."""
    if not 0 <= max_restarts <= 3:
        raise ValueError('max_restarts must be between zero and three')
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    owner = (root / 'watchdog.lock').open('a')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    statefile = root / 'watchdog_state.json'
    previous = read(statefile, {})
    attempts = previous.get('attempts', [])
    observed_existing_since = previous.get('observed_existing_since')
    state = dict(schema='chapter3-bounded-dispatcher-watchdog-v1', pid=os.getpid(),
                 scope=str(root), command=command, max_restarts=max_restarts,
                 attempts=attempts, observed_existing_since=observed_existing_since,
                 scientific_workers_never_restarted=True)

    def record(status, **extra):
        state.update(status=status, updated=time.time(), **extra)
        write(statefile, state)

    try:
        while True:
            if queue_locked(root):
                if observed_existing_since is None:
                    observed_existing_since = time.time()
                record('watching_existing_dispatcher', observed_existing_since=observed_existing_since)
                time.sleep(poll_seconds)
                continue
            if complete(root):
                record('complete', automatic_extra_training=False)
                return state
            blocked = blocker(root)
            if blocked:
                record('blocked', **blocked)
                return state
            if observed_existing_since is not None:
                blocked = failure_blocker(root, observed_existing_since)
                if blocked:
                    record('blocked', **blocked)
                    return state
                observed_existing_since = None
                state['observed_existing_since'] = None
            if attempts:
                blocked = failure_blocker(root, attempts[-1]['started'])
                if blocked:
                    record('blocked', **blocked)
                    return state
                if attempts[-1].get('exitcode') == 0:
                    record('blocked', reason='dispatcher stopped cleanly with incomplete scope; no blind relaunch')
                    return state
            if len(attempts) >= max_restarts + 1:
                record('restart_budget_exhausted', reason='bounded engineering retries exhausted')
                return state
            log = root / 'logs/watchdog_dispatcher.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            attempt = dict(index=len(attempts), started=time.time())
            attempts.append(attempt)
            record('launching_dispatcher')
            # A competing launch can only win queue.lock; the dispatcher never
            # bypasses that lock and this supervisor never changes worker state.
            with log.open('ab') as stream:
                process = subprocess.Popen(command, cwd=(root / 'execution/source')
                    if (root / 'execution/source').is_dir() else root,
                    stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), close_fds=True)
                attempt['pid'] = process.pid
                record('supervising_dispatcher')
                while process.poll() is None:
                    time.sleep(poll_seconds)
                attempt.update(exitcode=process.returncode, finished=time.time())
                record('dispatcher_exited')
            # If another valid dispatcher won the queue lock, observe it rather
            # than treating its lock conflict as an engineering failure.
    finally:
        owner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-restarts', type=int, default=2, choices=range(4))
    args = parser.parse_args()
    result = supervise(ROOT, [PYTHON, str(ROOT / 'execution/dispatch.py')], max_restarts=args.max_restarts)
    write(ROOT/'watchdog_recovery_launch.json', dict(status=result['status'], command=None if result['status']=='complete' else [PYTHON, str(Path(__file__).resolve())], automatic_extra_training=False))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
