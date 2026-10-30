"""The optimization graph for one op: seed -> strategy nodes -> combine -> merge.

Every model-backed node runs its own propose/check loop in its own working directory with its own
proposer (separate contexts, so the first opinion written does not become everyone's opinion).
GPU checks are serialized behind one lock: model calls run in parallel, measurements never do,
both for timing fidelity and because the GPU is shared.
"""
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .graph import Event, Graph, GraphRun, Node
from .gates import decide
from .loop import run_loop
from .progress import Progress
from .strategies import STRATEGIES, plan
from .verify import Report

GAIN = 1.02          # a strategy counts as a gain only if it beats its starting point by 2%
SEED_PROMPT = ("Write a straightforward, correct first version of this kernel. Do NOT optimize yet: "
               "other steps will apply specific optimizations to your code afterwards.")


@dataclass(frozen=True)
class Cand:
    name: str
    code: str
    speedup: float
    gained: bool
    outcome: str
    attempts: int
    lineage: tuple = ()        # which steps' ideas this code actually contains
    base_speedup: float | None = None   # speed of the starting point this step improved on
    trail: tuple = ()          # (stage, summary, approach) per attempt: raw material for lessons


@dataclass(frozen=True)
class Final:
    code: str
    path: str
    speedup: float             # as measured when this candidate was produced
    confirmed_speedup: float | None     # re-measured by a fresh check at the end
    winner: str
    table: tuple               # (node, speedup, gained, outcome, attempts)
    cands: tuple = ()          # every node's Cand, for deriving lessons


class GuidedProposer:
    """Wraps any proposer: starts from `base_code` and carries the node's instruction in PROGRESS.md."""

    def __init__(self, inner, instruction: str, base_code: str | None, first_note: str = "", hints=()):
        self.inner, self.instruction, self.base_code, self.first_note = inner, instruction, base_code, first_note
        self.hints = list(hints)

    def propose(self, op, progress: Progress, feedback, last_code):
        for line in [self.instruction] + self.hints:
            if line not in progress.known:
                progress.known.append(line)
        if last_code is None and self.base_code is not None:
            last_code = self.base_code
            feedback = self.first_note or f"Apply the instruction under 'known' to this code."
        return self.inner.propose(op, progress, feedback, last_code)


def build_graph(op: str, strategy_names: list[str], proposer_factory: Callable[[], object],
                check_factory: Callable[[float], Callable[[str], Report]], workdir: Path,
                max_attempts: int = 2, workers: int = 3, log: Callable[[str], None] = print,
                on_event: Callable[[Event], None] | None = None,
                sweep: Callable[[str], Report] | None = None, reviewer=None, lessons=None) -> Graph:
    gpu = threading.Lock()

    def locked(check):
        def run(path):
            with gpu:
                return check(path)
        return run

    quiet = lambda s: None
    hints = lessons.hints(op) if lessons else []

    def cand_from(name, loop, base: Cand | None, merged: tuple = ()) -> Cand:
        best = loop.best
        atts = loop.progress.attempts
        trail = tuple((a.stage, a.summary, a.approach) for a in atts)
        bs = base.speedup if base else None
        if loop.best_path is None or best is None:
            if base is None:
                raise RuntimeError(f"{name}: no correct candidate after {len(atts)} attempts")
            return Cand(name, base.code, base.speedup, False, loop.outcome, len(atts), base.lineage, bs, trail)
        gained = base is None or best.speedup >= base.speedup * GAIN
        if not gained:
            return Cand(name, base.code, base.speedup, False, loop.outcome, len(atts), base.lineage, bs, trail)
        lineage = tuple(dict.fromkeys((base.lineage if base else ()) + merged + (name,)))
        return Cand(name, loop.best_path.read_text(), best.speedup, True, loop.outcome, len(atts), lineage, bs, trail)

    def seed(_):
        loop = run_loop(op, GuidedProposer(proposer_factory(), SEED_PROMPT, None, hints=hints), locked(check_factory(0.0)),
                        workdir / "seed", target=0.0, max_attempts=max(3, max_attempts), log=quiet)
        return cand_from("seed", loop, None)

    def best_start(inputs, dep_names):
        pool = [inputs["seed"]] + [inputs[d] for d in dep_names if d in inputs and inputs[d].gained]
        return max(pool, key=lambda c: c.speedup)

    def strategy_node(sname: str, dep_names: tuple):
        def fn(inputs):
            base = best_start(inputs, dep_names)
            target = base.speedup * GAIN
            n_attempts = lessons.attempts_for(op, sname, max_attempts) if lessons else max_attempts
            if n_attempts == 0:
                return Cand(sname, base.code, base.speedup, False, "skipped by lesson", 0, base.lineage,
                            base.speedup, ())
            note = (f"Starting point: the code above runs at {base.speedup:.2f}x vs the reference. Apply ONLY this "
                    f"strategy ({sname}): {STRATEGIES[sname].prompt} Keep it correct.")
            loop = run_loop(op, GuidedProposer(proposer_factory(), f"Strategy for this step: {STRATEGIES[sname].prompt}",
                                               base.code, note, hints=hints),
                            locked(check_factory(target)), workdir / sname, target=target,
                            max_attempts=n_attempts, log=quiet)
            return cand_from(sname, loop, base)
        return fn

    def combine(inputs):
        """Merge only candidates that none of the others already contains (by lineage)."""
        gained = [c for k, c in inputs.items() if k in strategy_names and c.gained]
        subsumed = lambda c: any(o is not c and set(c.lineage) < set(o.lineage) for o in gained)
        cands = sorted((c for c in gained if not subsumed(c)), key=lambda c: -c.speedup)
        base0 = inputs.get("seed") or cands[0]
        if len(cands) < 2:
            return Cand("combine", base0.code, base0.speedup, False,
                        "skipped (fewer than 2 independent gains to combine)", 0, base0.lineage, base0.speedup)
        base, others = cands[0], cands[1:]
        listing = "\n\n".join(f"# variant with {'+'.join(c.lineage)} ({c.speedup:.2f}x)\n{c.code}" for c in others)
        note = (f"The code above is the best variant ({base.speedup:.2f}x). Merge the ideas of these other verified "
                f"variants into it, without breaking correctness:\n{listing}")
        target = base.speedup * GAIN
        loop = run_loop(op, GuidedProposer(proposer_factory(), "Combine the optimizations from several variants.",
                                           base.code, note, hints=hints),
                        locked(check_factory(target)), workdir / "combine", target=target,
                        max_attempts=max_attempts, log=quiet)
        merged = tuple(x for c in others for x in c.lineage)
        return cand_from("combine", loop, base, merged)

    def merge(inputs):
        """Plain code, no model: pick the fastest verified candidate, then re-measure it once."""
        cands = list(inputs.values())
        best = max(cands, key=lambda c: c.speedup)
        out = workdir / "final.py"
        out.write_text(best.code)
        confirmed = None
        try:
            rep = locked(check_factory(0.0))(str(out))
            confirmed = rep.speedup if rep.stage in ("ok", "speed") else None
        except Exception:
            pass
        table = tuple((c.name, round(c.speedup, 2), c.gained, c.outcome, c.attempts) for c in cands)
        return Final(best.code, str(out), best.speedup, confirmed, best.name, table, tuple(cands))

    deps = plan(strategy_names)
    nodes = [Node("seed", seed, retries=1)]
    for s, d in deps.items():
        nodes.append(Node(s, strategy_node(s, d), deps=("seed",) + d, retries=1, tolerant=True))
    nodes.append(Node("combine", combine, deps=("seed",) + tuple(deps), tolerant=True))
    nodes.append(Node("merge", merge, deps=("seed",) + tuple(deps) + ("combine",), tolerant=True))

    def gate(inputs):
        final = inputs["merge"]
        return decide(op, final.code, final.path, locked(sweep) if sweep else None, reviewer, speedup=final.speedup)

    nodes.append(Node("gate", gate, deps=("merge",)))
    return Graph(nodes, workers=workers, on_event=on_event)


def run_pipeline(*args, **kw) -> tuple[GraphRun, Final | None]:
    run = build_graph(*args, **kw).run()
    return run, run.results.get("merge")
