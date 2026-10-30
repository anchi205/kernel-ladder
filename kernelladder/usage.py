"""A persistent record of shared-GPU time, so 'how much have we used' has an answer across sessions.

One JSON line per remote run: when, how long it held the GPU (wall seconds), and what it was for.
Default location is ~/.kernelladder/gpu_usage.jsonl (override with KL_USAGE_LOG).
"""
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_LOG = Path(os.environ.get("KL_USAGE_LOG", Path.home() / ".kernelladder" / "gpu_usage.jsonl"))


def record(path: Path, seconds: float, tag: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"ts": datetime.now().astimezone().isoformat(timespec="seconds"),
                            "seconds": round(seconds, 2), "tag": tag}) + "\n")


def _rows(path: Path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
            row["dt"] = datetime.fromisoformat(row["ts"])
            yield row
        except (ValueError, KeyError):
            continue            # a torn or hand-edited line must not break accounting


def seconds_since(path: Path, since: datetime) -> float:
    return sum(r["seconds"] for r in _rows(path) if r["dt"] >= since)


def seconds_today(path: Path) -> float:
    now = datetime.now().astimezone()
    return seconds_since(path, now.replace(hour=0, minute=0, second=0, microsecond=0))


def summary(path: Path) -> str:
    rows = list(_rows(path))
    if not rows:
        return f"no GPU usage recorded yet ({path})"
    now = datetime.now().astimezone()
    start_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    windows = [("today", start_today), ("last 7 days", start_today - timedelta(days=6)), ("all time", datetime(1970, 1, 1, tzinfo=timezone.utc))]
    lines = [f"GPU time held on the shared box ({path})"]
    for name, since in windows:
        sel = [r for r in rows if r["dt"] >= since]
        lines.append(f"  {name:<12}{sum(r['seconds'] for r in sel) / 60:>7.1f} min   {len(sel)} runs")
    by_tag = defaultdict(float)
    for r in rows:
        by_tag[r.get("tag", "").split(":")[0] or "?"] += r["seconds"]
    lines.append("  by kind     " + ", ".join(f"{k} {v / 60:.1f} min" for k, v in sorted(by_tag.items(), key=lambda kv: -kv[1])))
    return "\n".join(lines)
