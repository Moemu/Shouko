"""Stdlib-only cooperative training control; stdin commands are newline-delimited.

Only FLYBODY_RUN_ID launches read stdin. The reader requests changes, but only
safe_point acknowledges a pause and stops the active clock. No training state
is discarded, including CUDA memory. EOF does not finish or resume a run.
"""
import os
import sys
import threading
import time


class TrainingControl:
    def __init__(self, max_seconds, stream=None):
        self.max_seconds = float(max_seconds)
        self.run_id = os.environ.get('FLYBODY_RUN_ID')
        self._condition = threading.Condition()
        self._pause_requested = False
        self.finish_requested = False
        self._started = time.perf_counter()
        self._paused_total = 0.0
        self._paused_at = None
        self._reader = None
        if self.run_id is not None:
            self._reader = threading.Thread(target=self._read_loop,
                                            args=(sys.stdin if stream is None else stream,), daemon=True)
            self._reader.start()

    def _read_loop(self, stream):
        try:
            for line in stream:
                command = line.strip().lower()
                with self._condition:
                    if command == 'finish':
                        self.finish_requested = True
                        self._pause_requested = False
                    elif not self.finish_requested:
                        if command == 'pause':
                            self._pause_requested = True
                        elif command == 'resume':
                            self._pause_requested = False
                    self._condition.notify_all()
        except (OSError, ValueError):
            pass  # Closed supervisor pipe is not a finish request.

    def fields(self):
        """Consistent clocks plus Unix-time heartbeat for the supervisor."""
        with self._condition:
            now = time.perf_counter()
            paused = self._paused_total + (now - self._paused_at if self._paused_at is not None else 0)
            wall = now - self._started
            return dict(run_id=self.run_id, max_seconds=self.max_seconds,
                        active_elapsed=wall - paused, paused_seconds=paused,
                        wall_elapsed=wall, heartbeat_at=time.time(),
                        control_enabled=self.run_id is not None)

    def active_elapsed(self):
        return self.fields()['active_elapsed']

    def paused_seconds(self):
        return self.fields()['paused_seconds']

    def wall_elapsed(self):
        return self.fields()['wall_elapsed']

    def budget_exceeded(self):
        return self.active_elapsed() >= self.max_seconds

    def safe_point(self, write_status):
        """Block on pause; return False on finish, True otherwise.

        write_status accepts a phase string, or None to restore its last active
        phase after resuming. 'paused' is an acknowledgement at a safe boundary;
        it is refreshed every second while blocked. 'stopping' requests that
        the caller save its final checkpoint and publish 'stopped'. Budget
        checks belong to the caller, after this method returns.
        """
        with self._condition:
            if self._pause_requested and not self.finish_requested:
                self._paused_at = time.perf_counter()
                try:
                    while self._pause_requested and not self.finish_requested:
                        write_status('paused')
                        self._condition.wait(timeout=1.0)
                finally:
                    self._paused_total += time.perf_counter() - self._paused_at
                    self._paused_at = None
                if not self.finish_requested:
                    write_status(None)
            if self.finish_requested:
                write_status('stopping')
                return False
            return True
