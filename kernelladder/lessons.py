"""Lessons: what one run teaches the next, as constraints with evidence, not transcripts.

Two kinds, both derived by code from structured run data (no extra model calls):
  strategy  "on <arch>, for <family> ops, <strategy> gained on n of m ops tried"
  pitfall   "this failure signature recurred; this is what fixed it"
Each record keeps its contributions per run, so any run's lessons can be removed (revert) and
totals are always recomputed from the evidence, never edited by hand.

Lessons change the search, never the verdict: they set how many attempts a strategy gets and what
hints the proposer sees. The verifier and gates are unaffected, so a wrong lesson costs search time,
not correctness.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .strategies import STRATEGIES

FAMILY = {"softmax": "memory-bound", "rmsnorm": "memory-bound", "kv_quant": "memory-bound",
          "attention": "compute-bound", "int8_matmul": "compute-bound"}
MAX_RECORDS = 200
MAX_HINTS = 6


def signature(text: str) -> str:
    """Failure identity with numbers and addresses masked, short enough to match across runs."""
    return re.sub(r"[-+]?\d[\d.]*(e[-+]?\d+)?", "N", text.strip().splitlines()[0] if text.strip() else "")[:100]


@dataclass(frozen=True)
class LessonView:
    records: tuple
    arch: str

    def _strategy_contribs(self, op: str, strategy: str) -> list[dict]:
        fam = FAMILY[op]
        return [c for r in self.records if r["kind"] == "strategy" and r["arch"] == self.arch
                and r["family"] == fam and r["key"] == strategy for c in r["contribs"]]

    def attempts_for(self, op: str, strategy: str, default: int) -> int:
        """Graded, not absolute: one prior miss makes it a one-attempt probe, two misses skip it."""
        cs = self._strategy_contribs(op, strategy)
        gained = sum(1 for c in cs if c["gained"])
        if len(cs) >= 2 and gained == 0:
            return 0
        if len(cs) >= 1 and gained == 0:
            return min(1, default)
        return default

    def hints(self, op: str) -> list[str]:
        fam, lines = FAMILY[op], []
        for s in STRATEGIES:
            cs = self._strategy_contribs(op, s)
            if not cs:
                continue
            gained = [c for c in cs if c["gained"]]
            if gained:
                best = max(c["gain_x"] for c in gained)
                lines.append(f"Lesson ({self.arch}, {fam}): '{s}' gained on {len(gained)} of {len(cs)} op(s), up to {best:.2f}x.")
            else:
                lines.append(f"Lesson ({self.arch}, {fam}): '{s}' gained nothing on {len(cs)} op(s) tried.")
        pit = [r for r in self.records if r["kind"] == "pitfall" and r["arch"] == self.arch
               and r["family"] == fam and len(r["contribs"]) >= 2]
        for r in sorted(pit, key=lambda r: -len(r["contribs"]))[:3]:
            fix = r["contribs"][-1]["fix"][:160]
            lines.append(f"Lesson (recurring failure, seen {len(r['contribs'])}x): '{r['key']}'. What fixed it: {fix}")
        return lines[:MAX_HINTS]


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.records: list[dict] = json.loads(self.path.read_text()) if self.path.exists() else []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.records[-MAX_RECORDS:], indent=1))

    def view(self, arch: str) -> LessonView:
        return LessonView(tuple(self.records), arch)

    def _record(self, kind: str, arch: str, family: str, key: str) -> dict:
        for r in self.records:
            if (r["kind"], r["arch"], r["family"], r["key"]) == (kind, arch, family, key):
                return r
        r = {"kind": kind, "arch": arch, "family": family, "key": key, "contribs": []}
        self.records.append(r)
        return r

    def learn(self, op: str, arch: str, cands, run_id: str) -> int:
        """Derive lessons from one finished graph run (its Cand objects). Returns contributions added."""
        fam, added = FAMILY[op], 0
        for c in cands:
            if c.name in STRATEGIES and c.attempts > 0 and c.base_speedup:
                self._record("strategy", arch, fam, c.name)["contribs"].append(
                    {"run": run_id, "op": op, "gained": bool(c.gained),
                     "gain_x": round(c.speedup / c.base_speedup, 3)})
                added += 1
            trail = c.trail
            for a, b in zip(trail, trail[1:]):         # (stage, summary, approach) triples
                if a[0] in ("correctness", "load") and b[0] not in ("correctness", "load", "fence"):
                    self._record("pitfall", arch, fam, signature(a[1]))["contribs"].append(
                        {"run": run_id, "op": op, "fix": b[2]})
                    added += 1
        self.save()
        return added

    def drop_run(self, run_id: str) -> int:
        n = 0
        for r in self.records:
            before = len(r["contribs"])
            r["contribs"] = [c for c in r["contribs"] if c["run"] != run_id]
            n += before - len(r["contribs"])
        self.records = [r for r in self.records if r["contribs"]]
        self.save()
        return n

    def describe(self) -> str:
        if not self.records:
            return "no lessons stored"
        lines = []
        for r in self.records:
            runs = sorted({c["run"] for c in r["contribs"]})
            if r["kind"] == "strategy":
                g = sum(c["gained"] for c in r["contribs"])
                lines.append(f"[{r['arch']} {r['family']}] strategy '{r['key']}': gained {g}/{len(r['contribs'])}   runs {runs}")
            else:
                lines.append(f"[{r['arch']} {r['family']}] pitfall x{len(r['contribs'])}: {r['key']}   runs {runs}")
        return "\n".join(lines)
