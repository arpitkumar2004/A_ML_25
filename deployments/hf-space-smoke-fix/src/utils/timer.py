# src/utils/timer.py
import time
from contextlib import contextmanager
from typing import Optional


@contextmanager
def timer(name: Optional[str] = None, logger=None):
    """
    Context manager for timing code blocks.
    Usage:
        with timer("fit", logger=logger):
            model.fit(...)
    """
    t0 = time.time()
    yield
    t1 = time.time()
    message = f"[TIMER] {name or 'block'} took {t1 - t0:.2f}s"
    if logger:
        logger.info(message)
    else:
        print(message)
        
class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print(f"Elapsed time: {self.interval:.2f} sec")
