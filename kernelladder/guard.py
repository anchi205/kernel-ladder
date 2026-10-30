"""Shared-GPU etiquette: look before using, meter what we use, and tell the human.

The box is someone else's. Before every remote run we check that nobody else is using the GPU,
we keep a running total of GPU wall time, and we alert on the Mac when either gets large.
Defaults (override with env): busy above 25% util, 5 min budget, hard stop at 2x budget.
"""
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import usage


class GuardStop(RuntimeError):
    """The loop must stop: continuing would hog a shared GPU."""


class GpuBusy(GuardStop):
    pass


class GpuBudget(GuardStop):
    pass


def alert(msg: str) -> None:
    print(f"\n!! KERNELLADDER ALERT: {msg}\n", file=sys.stderr, flush=True)
    if sys.platform == "darwin":
        safe = msg.replace("\\", " ").replace('"', "'")
        subprocess.run(["osascript", "-e", f'display notification "{safe}" with title "KernelLadder" sound name "Glass"'],
                       capture_output=True, timeout=10)


def _env(name: str, default: str) -> float:
    return float(os.environ.get(name, default))


@dataclass
class Guard:
    busy_util: float = field(default_factory=lambda: _env("KL_BUSY_UTIL", "25"))        # % GPU util
    busy_wait_s: float = field(default_factory=lambda: _env("KL_BUSY_WAIT_S", "120"))   # wait this long, then stop
    budget_s: float = field(default_factory=lambda: _env("KL_GPU_BUDGET_MIN", "5") * 60)
    daily_s: float = field(default_factory=lambda: _env("KL_GPU_DAILY_MIN", "30") * 60)   # across all sessions today
    log: Path | None = None             # persistent usage log; None disables (tests)
    notify: Callable[[str], None] = alert
    sleep: Callable[[float], None] = time.sleep
    used_s: float = 0.0
    _over: bool = False
    _daily_alerted: bool = False
    _busy_alerted: bool = False

    def before(self, util_fn: Callable[[], float]) -> None:
        if self.used_s >= 2 * self.budget_s:
            raise GpuBudget(f"used {self.used_s / 60:.1f} min of GPU time, over twice the {self.budget_s / 60:g} min budget")
        waited = 0.0
        while (u := util_fn()) > self.busy_util:
            if not self._busy_alerted:
                self.notify(f"GPU is busy ({u:.0f}% util, someone else is using it). Waiting up to {self.busy_wait_s:g}s.")
                self._busy_alerted = True
            if waited >= self.busy_wait_s:
                raise GpuBusy(f"GPU still at {u:.0f}% util after waiting {waited:g}s; not running on top of someone else")
            self.sleep(10)
            waited += 10
        self._busy_alerted = False

    def after(self, elapsed_s: float, tag: str = "") -> None:
        self.used_s += elapsed_s
        if self.log is not None:
            usage.record(self.log, elapsed_s, tag)
            today = usage.seconds_today(self.log)
            if today >= self.daily_s and not self._daily_alerted:
                self._daily_alerted = True
                self.notify(f"Today's total GPU time on the shared box is {today / 60:.0f} min "
                            f"(daily alert at {self.daily_s / 60:g} min), across all KernelLadder sessions.")
        if self.used_s >= self.budget_s and not self._over:
            self._over = True
            self.notify(f"KernelLadder has used {self.used_s / 60:.1f} min of GPU time on the shared box "
                        f"(budget {self.budget_s / 60:g} min). It will stop at {2 * self.budget_s / 60:g} min.")


GUARD = Guard(log=usage.DEFAULT_LOG)
