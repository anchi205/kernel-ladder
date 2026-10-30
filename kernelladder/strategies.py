"""Optimization strategies as transformations, and the scheduler that orders them.

Each strategy declares what it touches. Two strategies that touch disjoint things commute: applying
them in either order gives the same kind of result, so they can run in parallel from the same
starting point. Strategies that overlap conflict: they are ordered (earlier `order` first), and the
later one starts from the earlier one's output. This is the early-start test applied to
transformations: an edge exists only where starting B before A finishes would actually break.
"""
from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class Strategy:
    name: str
    prompt: str
    touches: frozenset
    order: int
    after: tuple = ()           # explicit "must follow these", beyond what overlap implies


STRATEGIES = {s.name: s for s in [
    Strategy("fuse", "Fuse separate kernels and intermediate tensors into as few kernels as possible, so "
             "intermediate results never round-trip through global memory.", frozenset({"fusion"}), 0),
    Strategy("vectorize", "Make global-memory access efficient: coalesced, aligned, vectorized loads and stores; "
             "avoid redundant reads.", frozenset({"memory_access"}), 1),
    Strategy("reduce", "Improve reductions: do row/tile reductions inside the kernel with cheap cooperative "
             "reductions (e.g. tl.sum / warp-level), keep partial sums in registers, avoid atomics.",
             frozenset({"reduction"}), 1),
    Strategy("tile", "Tile the computation so each block reuses data from registers or shared memory instead of "
             "re-reading global memory; pick tile shapes that fit the hardware.",
             frozenset({"memory_access", "reuse"}), 2),
    Strategy("launch", "Tune launch parameters only: block/tile sizes, num_warps, num_stages, grid shape. Do not "
             "change the algorithm.", frozenset({"launch_config"}), 9,
             after=("fuse", "vectorize", "reduce", "tile")),
]}

DEFAULT_STRATEGIES = {
    "softmax": ("fuse", "reduce", "vectorize", "launch"),
    "rmsnorm": ("fuse", "reduce", "vectorize", "launch"),
    "kv_quant": ("fuse", "reduce", "vectorize", "launch"),
    "attention": ("fuse", "tile", "reduce", "launch"),
    "int8_matmul": ("tile", "vectorize", "launch"),
}


def commute(a: Strategy, b: Strategy) -> bool:
    return not (a.touches & b.touches) and a.name not in b.after and b.name not in a.after


def plan(names: list[str]) -> dict[str, tuple[str, ...]]:
    """Return {strategy: the strategies it must wait for}, with redundant edges removed."""
    strategies = sorted((STRATEGIES[n] for n in names), key=lambda s: (s.order, s.name))
    deps = {s.name: set() for s in strategies}
    for a, b in combinations(strategies, 2):         # a sorts before b
        if not commute(a, b):
            deps[b.name].add(a.name)

    def reachable(start: str, goal: str, skip: str) -> bool:
        stack, seen = [d for d in deps[start] if d != skip], set()
        while stack:
            cur = stack.pop()
            if cur == goal:
                return True
            if cur not in seen:
                seen.add(cur)
                stack += list(deps[cur])
        return False

    for n in deps:                                   # transitive reduction
        for d in list(deps[n]):
            if reachable(n, d, skip=d):
                deps[n].discard(d)
    return {n: tuple(sorted(d)) for n, d in deps.items()}


def plan_text(names: list[str]) -> str:
    deps = plan(names)
    lines = ["seed -> " + ", ".join(n for n, d in deps.items() if not d) + "   (run in parallel)"]
    for n, d in deps.items():
        if d:
            lines.append(f"{' + '.join(d)} -> {n}   (conflict: {n} starts from the result of {' and '.join(d)})")
    lines.append("all strategies -> combine -> merge (plain code, picks the best verified result)")
    return "\n".join(lines)
