"""The loop: propose -> check -> feedback -> repeat, with exits.

The check is the only authority on "done"; the proposer never grades itself.
Exits: green | same failure twice | speed plateau | attempt budget.
"""
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from .guard import GuardStop
from .progress import Attempt, Progress
from .verify import Report

PLATEAU_GAIN = 1.02   # a speed attempt must beat the best so far by 2% to count as progress


class Proposer(Protocol):
    def propose(self, op: str, progress: Progress, feedback: str | None, last_code: str | None) -> tuple[str, str]:
        """Return (approach, full candidate source)."""


@dataclass
class Result:
    outcome: str                  # green | stuck | plateau | budget
    best: Attempt | None
    best_path: Path | None
    progress: Progress


def signature(report: Report) -> str:
    """Failure identity: stage plus the SAW text with numbers masked, so 'same error' survives noise."""
    return report.stage + ":" + re.sub(r"[-+]?\d[\d.]*(e[-+]?\d+)?", "N", report.saw)


def run_loop(op: str, proposer: Proposer, check: Callable[[str], Report], workdir: Path,
             target: float = 1.0, max_attempts: int = 8, log: Callable[[str], None] = print) -> Result:
    workdir.mkdir(parents=True, exist_ok=True)
    progress = Progress(goal=f"candidate for '{op}' that matches the reference and runs >= {target:.2f}x faster",
                        never=["calling the op's own library function", "importing kernelladder",
                               "editing anything but the candidate file"])
    feedback, last_code = None, None
    seen: dict[str, int] = {}
    best_speed = 0.0
    stalls = 0
    best_path = None
    outcome = "budget"

    for n in range(1, max_attempts + 1):
        approach, code = proposer.propose(op, progress, feedback, last_code)
        path = workdir / f"attempt_{n:02d}.py"
        path.write_text(code)
        try:
            report = check(str(path))
        except GuardStop as e:
            log(f"stopped: {e}")
            outcome = "gpu_guard"
            break
        last_code = code
        log(f"attempt {n}: {approach}\n  -> {report.feedback()}")

        summary = "green" if report.green else report.saw.splitlines()[0][:160]
        progress.attempts.append(Attempt(n, approach, report.stage, summary, report.speedup))
        if report.speedup is not None and report.speedup > best_speed:
            best_path = path
        progress.save(workdir / "PROGRESS.md")

        if report.green:
            outcome = "green"
            break

        sig = signature(report)
        seen[sig] = seen.get(sig, 0) + 1
        if report.stage != "speed" and seen[sig] >= 2:
            outcome = "stuck"      # same failure twice: more attempts will not help, escalate
            break
        if report.stage == "speed":
            if report.speedup > best_speed * PLATEAU_GAIN:
                stalls = 0
            else:
                stalls += 1
            if stalls >= 2:
                outcome = "plateau"
                break
        if report.speedup is not None:
            best_speed = max(best_speed, report.speedup)
        feedback = report.feedback(progress.dead_ends())

    if best_path:
        shutil.copy(best_path, workdir / "best.py")
        best_path = workdir / "best.py"
    return Result(outcome, progress.best, best_path, progress)
