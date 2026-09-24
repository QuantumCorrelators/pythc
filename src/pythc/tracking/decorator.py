import functools
import logging
import time

from pythc import observe

logger = logging.getLogger()


def measure_time(func=None, *, name=None):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # observe.measure is a no-op without an active run, so no guard needed
        with observe.measure(name or func.__name__):
            return func(*args, **kwargs)

    return wrapper


class MeasureBlock:
    def __init__(self, name):
        self.name = name
        self.start_time = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time
        observe.log_metric(f"{self.name}_time", duration)
        logger.info(f"measureblock {self.name} took {duration} seconds")
