import json
import logging
import os
import platform
import socket
import time
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import git
import numpy as np
import psutil
import pymongo
from bson import Binary
from pyinstrument import Profiler
from pymongo import errors

from pythc.lib import cpu_count
from pythc.lib import latest_tag

# Global connection handling (Crucial for performance)
# We initialize this lazily so we don't connect until we actually save.
_MONGO_CLIENT = None

logger = logging.getLogger()

def clean_value(v):
    if isinstance(v, (int, float, str, bool, type(None), np.number, Binary)):
        return v

    if isinstance(v, (list, tuple, np.ndarray)):
        return [clean_value(item) for item in v]

    if hasattr(v, 'name') and isinstance(v.name, str):
        return v.name

    s = str(v)

    if s.startswith("<") and "object at 0x" in s:
        return v.__class__.__name__  # e.g., returns "Mole" or "DFRHF"

    return s


def get_mongo_collection():
    global _MONGO_CLIENT
    if _MONGO_CLIENT is None:
        mongo_url = os.environ.get("MONGO_URL")
        if not mongo_url:
            raise ValueError("MONGO_URL env var not set")

        _MONGO_CLIENT = pymongo.MongoClient(mongo_url)


    collection = _MONGO_CLIENT.pythc[latest_tag()]

    if not collection.list_indexes().to_list():
        collection.create_index(["dimensions", "metrics.status"], unique=True)

    return collection


type Backend = Literal['mongodb', 'file']

@dataclass(init=True)
class Checkpoint:
    name: str
    timestamp: datetime
    diff: float

    def encode(self):
        return {'name': self.name, 'timestamp': self.timestamp, 'diff': self.diff}

class ExperimentRun:
    _active = None


    def __init__(self, name, params, add_meta=True, backend: Backend ='file', checkpoints=True):
        self.name = name
        self.timestamp = None
        self.duration = 0
        self.params = params
        self.metrics = {}
        self.error = None
        self.add_meta = add_meta
        self.backend = backend
        self.checkpoints = []
        self.checkpoint_enabled=checkpoints
        self.profile = None


    @classmethod
    def get_active(cls):
        return cls._active

    @classmethod
    def exists(cls, params, backend: Backend = 'file') -> bool:
        if backend == 'mongodb':
            collection = get_mongo_collection()
            query = {f"dimensions.{k}": clean_value(v) for k, v in params.items() if k not in ['chkfile']}|{'metrics.status': "SUCCESS"}
            result = collection.find_one(query, {"_id": 1})
            exists = result is not None

            return exists

        return False

    def log_metric(self, key, value):
        self.metrics[key] = value

    def increase_metric(self,key, diff=1):
        if key not in self.metrics:
            self.metrics[key] = diff
        else:
            self.metrics[key] += diff

    def build(self, cls):
        # Check if the class knows how to build itself from config
        if hasattr(cls, 'from_config'):
            return cls.from_config(self.params)

        # Fallback for standard classes (optional)
        return cls(**self.params)

    @contextmanager
    def measure(self, name):
        """
        Context manager to measure a specific block of code.
        Usage: with run.measure("my_block_name"): ...
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.log_metric(f"{name}_time", dt)

    # --- Context Manager Protocol ---
    def __enter__(self):
        ExperimentRun._active = self
        self.timestamp = datetime.now()
        self.start_time = time.perf_counter()
        self.checkpoints = [Checkpoint('init', self.timestamp, 0)]

        logger.info(f"starting experiment run with parameters:\n {"\n".join([f"\t{k}={clean_value(v)}" for k, v in self.params.items()])}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.perf_counter() - self.start_time

        if exc_type:
            self.error = str(exc_val)
            self.metrics['status'] = "FAILED"
            logger.error(self.error)
        else:
            self.metrics['status'] = "SUCCESS"

        self.save()

        ExperimentRun._active = None

    def checkpoint(self, name: str):
        if not self.checkpoint_enabled :
            return

        last = self.checkpoints[-1]
        now = datetime.now()
        diff = now - last.timestamp
        self.checkpoints.append(
            Checkpoint(name, now, diff.total_seconds())
        )

    def to_row(self):
        """
        Flattens this object into a single dictionary for a DataFrame.
        Prefixes keys to avoid collisions.
        """
        row = {
            "meta": {
                'run_name': self.name,
                'timestamp': self.timestamp,
                'error_msg': self.error
            },
            "dimensions": {},
            "metrics": {},
            "checkpoints": [c.encode() for c in self.checkpoints],
            "profile_html": self.profile
        }

        if self.add_meta:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
            row["meta"]["git_commit_hash"]=sha
            row["meta"]["version"]=latest_tag()
            row["meta"]["cpu_count"]=cpu_count()
            row["meta"]["memory"]=psutil.virtual_memory().total
            row["meta"]["hostname"]=socket.gethostname()
            row["meta"]["cpu"]=platform.processor()

        self.log_metric("total_duration", self.duration)

        for k, v in self.params.items():
            row["dimensions"][k] = clean_value(v)

        for k, v in self.metrics.items():
            row["metrics"][k] = clean_value(v)

        return row

    def add_profile(self, prof: Profiler):
        self.profile = Binary(zlib.compress(prof.output_html().encode("utf-8")))


    def save(self, file_path: str = "experiments.jsonl"):
        if self.backend == 'mongodb':
            collection = get_mongo_collection()
            try:
                collection.insert_one(self.to_row())
                logger.info(f"Run {self.name} saved to MongoDB")
            except errors.DuplicateKeyError:
                logger.warning(f"Run already exists in DB! Skipping persist ...")

        elif self.backend == 'file':
            self._save_to_jsonl(file_path)

    def _save_to_jsonl(self, file_path: str):
        row = self.to_row()

        if "profile_html" in row:
            row["profile_html"] = "<binary_data_omitted>"

        with open(file_path, mode='a', encoding='utf-8') as f:
            f.write(json.dumps(row, default=str) + "\n")

        logger.info(f"Run {self.name} saved to JSONL ({file_path})")

