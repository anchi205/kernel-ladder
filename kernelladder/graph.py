"""A small dependency-graph executor.

NODE   one bounded job: reads its dependencies' results, returns one result.
EDGE   a real dependency: this node cannot start before that one is done.
STATE  results; once a node finishes its result is frozen (nodes get a read-only mapping).

A node starts the moment its own dependencies finish, not when a whole "level" finishes.
A failed node reruns alone (its retries); every finished result stays exactly as it was.
"""
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class Node:
    name: str
    fn: Callable[[Mapping[str, Any]], Any]
    deps: tuple[str, ...] = ()
    retries: int = 0
    tolerant: bool = False      # run even if some deps failed, seeing only the ones that succeeded


@dataclass(frozen=True)
class Event:
    t: float
    node: str
    kind: str                   # start | retry | done | failed | skipped
    detail: str = ""


@dataclass
class GraphRun:
    results: Mapping[str, Any]
    status: dict[str, str]      # done | failed | skipped
    errors: dict[str, str]
    events: list[Event]
    wall_s: float


def validate(nodes: list[Node]) -> None:
    names = [n.name for n in nodes]
    if len(set(names)) != len(names):
        raise ValueError("duplicate node names")
    for n in nodes:
        for d in n.deps:
            if d not in names:
                raise ValueError(f"node {n.name!r} depends on unknown node {d!r}")
    indeg = {n.name: len(n.deps) for n in nodes}
    ready = [k for k, v in indeg.items() if v == 0]
    seen = 0
    while ready:
        cur = ready.pop()
        seen += 1
        for n in nodes:
            if cur in n.deps:
                indeg[n.name] -= 1
                if indeg[n.name] == 0:
                    ready.append(n.name)
    if seen != len(nodes):
        raise ValueError("graph has a cycle")


class Graph:
    def __init__(self, nodes: list[Node], workers: int = 4, on_event: Callable[[Event], None] | None = None):
        validate(nodes)
        self.nodes, self.workers, self.on_event = nodes, workers, on_event
        self._lock = threading.Lock()
        self._events: list[Event] = []
        self._t0 = 0.0

    def _emit(self, node: str, kind: str, detail: str = "") -> None:
        ev = Event(time.monotonic() - self._t0, node, kind, detail)
        with self._lock:
            self._events.append(ev)
        if self.on_event:
            self.on_event(ev)

    def _run_node(self, node: Node, inputs: Mapping[str, Any]) -> Any:
        for attempt in range(node.retries + 1):
            try:
                return node.fn(inputs)
            except Exception as e:
                if attempt == node.retries:
                    raise
                self._emit(node.name, "retry", f"{type(e).__name__}: {e}")

    def run(self) -> GraphRun:
        self._t0 = time.monotonic()
        pending = {n.name: n for n in self.nodes}
        running: dict[Any, str] = {}
        results: dict[str, Any] = {}
        status: dict[str, str] = {}
        errors: dict[str, str] = {}
        with ThreadPoolExecutor(self.workers) as pool:
            while pending or running:
                progressed = False
                for name, node in list(pending.items()):
                    if any(d not in status for d in node.deps):
                        continue
                    bad = [d for d in node.deps if status[d] != "done"]
                    del pending[name]
                    progressed = True
                    if bad and not node.tolerant:
                        status[name] = "skipped"
                        self._emit(name, "skipped", f"dependency did not finish: {', '.join(bad)}")
                        continue
                    inputs = MappingProxyType({d: results[d] for d in node.deps if status[d] == "done"})
                    running[pool.submit(self._run_node, node, inputs)] = name
                    self._emit(name, "start")
                if not running:
                    if not progressed:
                        break
                    continue
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for fut in done:
                    name = running.pop(fut)
                    try:
                        results[name] = fut.result()
                        status[name] = "done"
                        self._emit(name, "done")
                    except Exception as e:
                        status[name] = "failed"
                        errors[name] = f"{type(e).__name__}: {e}"
                        self._emit(name, "failed", errors[name])
        return GraphRun(MappingProxyType(results), status, errors, self._events, time.monotonic() - self._t0)
