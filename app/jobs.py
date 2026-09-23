"""Tiny in-process job queue: video work runs on a thread pool and the UI polls for status."""
from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"  # queued | running | done | error
    progress: float = 0.0
    message: str = "Waiting to start"
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def update(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)

    def to_dict(self) -> dict:
        return asdict(self)


class JobStore:
    def __init__(self, workers: int = 2):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=workers)

    def submit(self, kind: str, fn: Callable[[Job], dict]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind)
        with self._lock:
            self._jobs[job.id] = job

        def run() -> None:
            job.update(status="running", message="Starting")
            try:
                result = fn(job)
                job.update(status="done", progress=1.0, message="Done", result=result)
            except Exception as exc:  # surfaced to the UI
                traceback.print_exc()
                job.update(status="error", error=str(exc) or exc.__class__.__name__, message="Failed")

        self._pool.submit(run)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)
