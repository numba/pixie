import functools
import threading

# NOTE: This is based on:
# https://github.com/numba/numba/blob/04ebc63fe1dd1efd5a68cc9caf8f245404d99fa7/numba/core/compiler_lock.py

class _CompilerLock(object):
    """Lock for the preventing multiple compiler executions"""
    def __init__(self):
        self._lock = threading.RLock()

    def acquire(self):
        self._lock.acquire()

    def release(self):
        self._lock.release()

    def __enter__(self):
        self.acquire()

    def __exit__(self, exc_val, exc_type, traceback):
        self.release()

    def is_locked(self):
        is_owned = getattr(self._lock, '_is_owned')
        if not callable(is_owned):
            is_owned = self._is_owned
        return is_owned()

    def __call__(self, func):
        @functools.wraps(func)
        def _acquire_compile_lock(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return _acquire_compile_lock

    def _is_owned(self):
        # This method is borrowed from threading.Condition.
        # Return True if lock is owned by current_thread.
        # This method is called only if _lock doesn't have _is_owned().
        if self._lock.acquire(0):
            self._lock.release()
            return False
        else:
            return True


compiler_lock = _CompilerLock()
