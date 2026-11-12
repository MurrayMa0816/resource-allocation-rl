"""Recover only this new controller; never retries scientific workers."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from controller import BOOT, identity, live, write

ROOT = Path(__file__).resolve().parents[1]


def main():
    lock = (ROOT/'watchdog.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(ROOT/'watchdog_pid.json', dict(pid=os.getpid(), boot_id=BOOT, **identity(os.getpid())))
    restarts = 0
    child = None
    while True:
        if (ROOT/'completion_receipt.json').exists():
            write(ROOT/'watchdog_status.json', dict(status='complete', restarts=restarts))
            return
        pidfile = ROOT/'queue_pid.json'
        record = json.loads(pidfile.read_text()) if pidfile.exists() else None
        if record and 'command' not in record:
            record = dict(record, command=record['argv'])
        running = live(record) if record else False
        if running is not False:
            time.sleep(5)
            continue
        if child is not None:
            code = child.poll()
            if code is None:
                time.sleep(2)
                continue
        if restarts >= 3:
            write(ROOT/'watchdog_status.json', dict(status='controller_restart_limit', restarts=restarts))
            return
        stream = (ROOT/'logs/controller.log').open('ab', buffering=0)
        child = subprocess.Popen([sys.executable, str(ROOT/'execution/controller.py')],
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        stream.close()
        restarts += 1
        write(ROOT/'watchdog_status.json', dict(status='controller_started', pid=child.pid, restarts=restarts))
        time.sleep(5)


if __name__ == '__main__':
    main()
