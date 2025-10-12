import time
from contextlib import contextmanager

@contextmanager
def timer(name="block"):
    t0 = time.time()
    yield
    print(f"[TIMER] {name} elapsed {time.time() - t0:.2f}s")
