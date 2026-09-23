"""Bounded isolated child jobs with retained timeout receipts, no shell."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime, timezone


class InspectionDeadline(TimeoutError):
    pass


def inspect_with_deadline(function, *args, timeout_s=1800):
    """Bound one complete source/stage inspection on Unix's main thread."""
    def expired(signum, frame):
        raise InspectionDeadline(f'Inspection exceeded {timeout_s} seconds')
    previous = signal.signal(signal.SIGALRM, expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        return function(*args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if previous_timer[0]:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def run(command, *, stdout, stderr=subprocess.STDOUT, timeout=900, **kwargs):
    """subprocess.run-compatible subset; timeout cannot silently discard results.

    A child owns a new session. Termination targets only that child process group,
    never another trial or a shared server. The parent evaluator retains logs and
    classifies missing native output as INCONCLUSIVE. Global slots are the outer
    broker's responsibility, not implemented by bypassable author-side claims.
    """
    if kwargs:
        raise TypeError('Unsupported process options: ' + ', '.join(kwargs))
    if isinstance(command, str):
        raise TypeError('A structured argument list is required')
    started = datetime.now(timezone.utc).isoformat()
    tick = time.monotonic()
    proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
    timed_out = False
    try:
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            code = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            code = proc.wait(timeout=5)
    receipt = {'schema_version': 1, 'started_at': started,
               'completed_at': datetime.now(timezone.utc).isoformat(),
               'elapsed_s': time.monotonic() - tick, 'timeout_s': timeout,
               'timed_out': timed_out, 'returncode': code,
               'disposition': 'INCONCLUSIVE' if timed_out else 'completed_process_only'}
    if hasattr(stdout, 'name'):
        Path(str(stdout.name) + '.process.json').write_text(json.dumps(receipt, indent=2) + '\n')
    result = subprocess.CompletedProcess(command, code)
    result.timed_out = timed_out
    return result
