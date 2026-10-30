"""Benchmark with a held-out split: does learning from some ops help on ops it never saw?

  train ops      run with no lessons; their runs produce the lessons
  held-out ops   each run twice: control (no lessons) and treatment (lessons from train ops only)

Control and treatment alternate per op so GPU contention drifts across both arms rather than one.
Each cell is a single run of a nondeterministic system: the report says so instead of implying more.
"""
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .lessons import FAMILY, Store


@dataclass
class Row:
    op: str
    arm: str                   # train | control | treatment
    speedup: float | None
    confirmed: float | None
    model_attempts: int        # attempts summed over every node: the search cost
    skipped: int               # strategies a lesson skipped outright
    wall_s: float
    gpu_s: float
    gate: str


def _row(op, arm, run, final, wall_s, gpu_s) -> Row:
    if final is None:
        return Row(op, arm, None, None, 0, 0, wall_s, gpu_s, "no result: " + "; ".join(run.errors.values())[:100])
    gate = run.results.get("gate")
    return Row(op, arm, round(final.speedup, 2), round(final.confirmed_speedup, 2) if final.confirmed_speedup else None,
               sum(c.attempts for c in final.cands), sum(1 for c in final.cands if c.outcome == "skipped by lesson"),
               round(wall_s, 1), round(gpu_s, 1), gate.decision if gate else "n/a")


def run_bench(train_ops: list[str], heldout_ops: list[str], arch: str, store_path: Path,
              run_one: Callable, gpu_seconds: Callable[[], float], log=print) -> list[Row]:
    """`run_one(op, lessons_view_or_None) -> (GraphRun, Final|None)`."""
    store = Store(store_path)
    rows: list[Row] = []

    def timed(op, arm, lessons):
        g0, t0 = gpu_seconds(), time.monotonic()
        run, final = run_one(op, lessons)
        row = _row(op, arm, run, final, time.monotonic() - t0, gpu_seconds() - g0)
        rows.append(row)
        log(f"[{arm:<9}] {op:<12} {row.speedup}x  attempts={row.model_attempts} skipped={row.skipped} "
            f"wall={row.wall_s:.0f}s gpu={row.gpu_s:.0f}s gate={row.gate}")
        return final

    for op in train_ops:
        final = timed(op, "train", None)
        if final is not None:
            n = store.learn(op, arch, final.cands, run_id=f"bench-{op}")
            log(f"           learned {n} lesson contribution(s) from {op}")

    view = store.view(arch)
    for i, op in enumerate(heldout_ops):
        order = [("control", None), ("treatment", view)] if i % 2 == 0 else [("treatment", view), ("control", None)]
        for arm, lessons in order:
            timed(op, arm, lessons)
    return rows


def report(rows: list[Row], arch: str, train_ops, heldout_ops) -> str:
    out = [f"# KernelLadder benchmark ({arch})", "",
           f"train ops: {', '.join(train_ops)} | held-out ops: {', '.join(heldout_ops)}", "",
           "| op | family | arm | speedup | re-measured | model attempts | skipped | wall s | GPU s | gate |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r.op} | {FAMILY[r.op]} | {r.arm} | {r.speedup}x | {r.confirmed}x | {r.model_attempts} | "
                   f"{r.skipped} | {r.wall_s:.0f} | {r.gpu_s:.0f} | {r.gate} |")
    out += ["", "## Held-out: with lessons vs without", ""]
    for op in heldout_ops:
        c = next((r for r in rows if r.op == op and r.arm == "control"), None)
        t = next((r for r in rows if r.op == op and r.arm == "treatment"), None)
        if c and t and c.speedup and t.speedup:
            out.append(f"- **{op}**: speedup {c.speedup}x -> {t.speedup}x, model attempts {c.model_attempts} -> "
                       f"{t.model_attempts}, GPU time {c.gpu_s:.0f}s -> {t.gpu_s:.0f}s, wall {c.wall_s:.0f}s -> {t.wall_s:.0f}s")
    out += ["", "> Each cell is **one run**. The proposer is nondeterministic and the GPU is shared, so differences",
            "> of a few percent in speedup or wall time are noise; attempt counts are the least noisy signal of",
            "> search cost. Treat this as a worked example of the method, not a statistic."]
    return "\n".join(out)


def save(rows: list[Row], text: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps([asdict(r) for r in rows], indent=1))
    (out / "report.md").write_text(text)
