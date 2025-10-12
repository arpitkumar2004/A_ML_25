"""Simple timer context manager."""
import time
class Timer:
    def __init__(self, name='timer'):
        self.name = name
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        print(f"{self.name} took {time.time()-self.start:.2f}s")

