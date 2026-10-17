"""Shared utilities: timing, logging, run-log data structure."""

import time
import functools
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DecisionRecord:
    """One entry in the agent's audit trail -- what stage, what decision,
    why (the reasoning text), and how long it took. This is what makes the
    final report able to *explain* the agent's autonomous choices instead of
    just showing final numbers."""
    stage: str
    decision: str
    reasoning: str
    duration_sec: float = 0.0
    extra: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class RunLog:
    """Central audit trail shared by every agent in the pipeline."""

    def __init__(self):
        self.records: list[DecisionRecord] = []
        self.start_time = time.time()

    def add(self, stage: str, decision: str, reasoning: str,
            duration_sec: float = 0.0, **extra):
        self.records.append(DecisionRecord(stage, decision, reasoning,
                                            duration_sec, extra))

    def record(self, stage: str, decision: str, reasoning: str,
               duration_sec: float = 0.0, **extra):
        """Alias for add to support log.record calls."""
        self.add(stage, decision, reasoning, duration_sec, **extra)

    def total_elapsed(self) -> float:
        return time.time() - self.start_time

    def as_markdown(self) -> str:
        lines = ["| Stage | Decision | Reasoning | Time (s) |",
                 "|---|---|---|---|"]
        for r in self.records:
            lines.append(
                f"| {r.stage} | {r.decision} | {r.reasoning} | {r.duration_sec:.2f} |"
            )
        return "\n".join(lines)


def timed_stage(stage_name: str):
    """Decorator that times an agent's `.run()` method and records it,
    provided the agent has a `self.log` (RunLog) attribute."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            t0 = time.time()
            result = fn(self, *args, **kwargs)
            dt = time.time() - t0
            if hasattr(self, "log") and self.log is not None:
                # last decision recorded during this call gets the duration
                # attached retroactively if none was set
                if self.log.records and self.log.records[-1].stage == stage_name:
                    self.log.records[-1].duration_sec = dt
                else:
                    self.log.add(stage_name, "completed", "no explicit decision logged", dt)
            return result
        return wrapper
    return decorator


def human_bytes(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"
