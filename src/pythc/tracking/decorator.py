import functools
import logging

import time

from pythc.tracking.experiment_run import ExperimentRun

logger = logging.getLogger()

def measure_time(func=None, *, name=None):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check for active run
        active_run = ExperimentRun.get_active()

        # If no experiment is running, just execute normally
        if not active_run:
            return func(*args, **kwargs)

        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            dt = time.perf_counter() - t0
            # Use provided name or default to function name
            metric_name = name or f"{func.__name__}_time"
            active_run.log_metric(metric_name, dt)

    return wrapper


class MeasureBlock:
    def __init__(self, name):
        self.name = name
        self.run = None
        self.start_time = 0

    def __enter__(self):
        # 1. Check if an experiment is actually running
        self.run = ExperimentRun.get_active()

        # 2. If yes, start the timer. If no, do nothing.
        if self.run:
            self.start_time = time.perf_counter()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 3. Only log if we actually started a timer
        duration = time.perf_counter() - self.start_time

        if self.run:
            self.run.log_metric(f"{self.name}_time", duration)
            logger.info(f"measureblock {self.name} took {duration} seconds")

