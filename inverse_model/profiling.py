import contextlib
import logging
import time
from collections import defaultdict

import torch


_enabled = False
_sync_cuda = True
_stats = defaultdict(lambda: {"total": 0.0, "count": 0})


def configure(config):
    """Configure lightweight timing instrumentation from the config."""
    global _enabled, _sync_cuda
    profiling_config = getattr(config, "profiling", None)
    _enabled = bool(getattr(profiling_config, "enabled", False))
    _sync_cuda = bool(getattr(profiling_config, "sync_cuda", True))


def enabled():
    return _enabled


def _sync_if_needed():
    if _enabled and _sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()


@contextlib.contextmanager
def record(name):
    """Record elapsed time for a named section when profiling is enabled."""
    if not _enabled:
        yield
        return

    _sync_if_needed()
    start = time.perf_counter()
    try:
        yield
    finally:
        _sync_if_needed()
        elapsed = time.perf_counter() - start
        stat = _stats[name]
        stat["total"] += elapsed
        stat["count"] += 1


def reset():
    _stats.clear()


def report(step=None, reset_after=True, top_k=None):
    """Log accumulated timing stats sorted by total time."""
    if not _enabled or not _stats:
        return

    items = sorted(_stats.items(), key=lambda item: item[1]["total"], reverse=True)
    if top_k is not None:
        items = items[:top_k]

    total_step_time = _stats.get("step_total", {}).get("total", 0.0)
    header = f"[profile] step={step}" if step is not None else "[profile]"
    logging.info("%s timing summary", header)
    for name, stat in items:
        total_ms = stat["total"] * 1000.0
        avg_ms = total_ms / max(stat["count"], 1)
        pct = ""
        if total_step_time > 0 and name != "step_total":
            pct = f", pct_of_step={stat['total'] / total_step_time * 100.0:.1f}%"
        logging.info(
            "[profile] %-40s total=%10.3f ms, avg=%9.3f ms, count=%6d%s",
            name,
            total_ms,
            avg_ms,
            stat["count"],
            pct,
        )

    if reset_after:
        reset()
