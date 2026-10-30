"""PROGRESS.md: memory that outlives the model's context window.

The loop owns this file, not the model: it is rewritten from structured state after
every attempt, so a compacted or restarted agent resumes from the file, not from a transcript.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Attempt:
    n: int
    approach: str
    stage: str          # fence | load | correctness | speed | ok
    summary: str        # one line: what happened
    speedup: float | None = None


@dataclass
class Progress:
    goal: str
    never: list[str] = field(default_factory=list)
    known: list[str] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def best(self) -> Attempt | None:
        scored = [a for a in self.attempts if a.speedup is not None]
        return max(scored, key=lambda a: a.speedup, default=None)

    def dead_ends(self) -> list[str]:
        return [f"#{a.n} {a.approach}" for a in self.attempts if a.stage not in ("ok",)
                and (a.speedup is None or a.speedup < 1.0)]

    def render(self) -> str:
        best = self.best
        lines = [f"# PROGRESS", "", f"goal      {self.goal}"]
        if best:
            lines.append(f"best      attempt #{best.n}, {best.speedup:.2f}x ({best.approach})")
        lines += ["", "## tried"]
        lines += [f"- #{a.n} [{a.stage}] {a.approach}: {a.summary}" for a in self.attempts] or ["- nothing yet"]
        lines += ["", "## known"] + [f"- {k}" for k in self.known] or [""]
        lines += ["", "## never"] + [f"- {n}" for n in self.never]
        return "\n".join(lines) + "\n"

    def save(self, path: Path):
        path.write_text(self.render())
