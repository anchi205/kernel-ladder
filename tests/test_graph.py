import threading
import time
from types import MappingProxyType

import pytest

from kernelladder.agent import ScriptedProposer
from kernelladder.graph import Graph, Node
from kernelladder.pipeline import run_pipeline
from kernelladder.strategies import commute, plan, STRATEGIES
from kernelladder.verify import Report


def test_independent_nodes_run_in_parallel_and_dependents_wait():
    order = []

    def slow(name):
        def fn(inputs):
            order.append(("start", name)); time.sleep(0.3); order.append(("end", name)); return name
        return fn

    nodes = [Node("a", slow("a")), Node("b", slow("b")),
             Node("c", lambda i: sorted(i), deps=("a", "b"))]
    run = Graph(nodes, workers=2).run()
    assert run.wall_s < 0.55                       # a and b overlapped
    assert run.results["c"] == ["a", "b"]
    assert order.index(("end", "a")) < len(order) and run.status == {"a": "done", "b": "done", "c": "done"}


def test_node_starts_when_its_own_deps_finish_not_when_a_level_finishes():
    t = {}
    nodes = [Node("fast", lambda i: time.sleep(0.05)), Node("slow", lambda i: time.sleep(0.5)),
             Node("after_fast", lambda i: t.setdefault("start", time.monotonic()), deps=("fast",))]
    t0 = time.monotonic()
    Graph(nodes, workers=3).run()
    assert t["start"] - t0 < 0.3                   # did not wait for 'slow'


def test_results_are_read_only_to_downstream_nodes():
    def tamper(inputs):
        with pytest.raises(TypeError):
            inputs["a"] = "changed"
        assert isinstance(inputs, MappingProxyType)
        return inputs["a"]
    run = Graph([Node("a", lambda i: "orig"), Node("b", tamper, deps=("a",))]).run()
    assert run.results["b"] == "orig"


def test_failed_node_reruns_alone_and_finished_work_is_not_redone():
    calls = {"good": 0, "flaky": 0}

    def good(i):
        calls["good"] += 1; return 1

    def flaky(i):
        calls["flaky"] += 1
        if calls["flaky"] < 2:
            raise RuntimeError("boom")
        return 2

    run = Graph([Node("good", good), Node("flaky", flaky, retries=1), Node("end", lambda i: sum(i.values()), deps=("good", "flaky"))]).run()
    assert calls == {"good": 1, "flaky": 2} and run.results["end"] == 3
    assert any(e.kind == "retry" for e in run.events)


def test_failure_skips_dependents_unless_tolerant():
    boom = Node("boom", lambda i: 1 / 0)
    run = Graph([boom, Node("strict", lambda i: 1, deps=("boom",)),
                 Node("tolerant", lambda i: dict(i), deps=("boom",), tolerant=True)]).run()
    assert run.status == {"boom": "failed", "strict": "skipped", "tolerant": "done"}
    assert run.results["tolerant"] == {}


def test_cycle_and_unknown_dep_rejected():
    with pytest.raises(ValueError):
        Graph([Node("a", lambda i: 1, deps=("b",)), Node("b", lambda i: 1, deps=("a",))])
    with pytest.raises(ValueError):
        Graph([Node("a", lambda i: 1, deps=("nope",))])


# ---- scheduler -----------------------------------------------------------------------------

def test_disjoint_strategies_commute_and_overlapping_ones_are_ordered():
    S = STRATEGIES
    assert commute(S["fuse"], S["reduce"]) and commute(S["fuse"], S["vectorize"])
    assert not commute(S["vectorize"], S["tile"])          # both touch memory_access
    assert not commute(S["launch"], S["fuse"])             # launch tuning must follow structure changes
    deps = plan(["fuse", "reduce", "vectorize", "tile", "launch"])
    assert deps["fuse"] == () and deps["reduce"] == () and deps["vectorize"] == ()
    assert deps["tile"] == ("vectorize",)
    assert set(deps["launch"]) == {"fuse", "reduce", "tile"}   # vectorize is implied via tile: edge removed


# ---- pipeline, with a fake check (no GPU, no model) ----------------------------------------

def test_pipeline_picks_best_verified_candidate_and_serializes_checks():
    speeds = {"SEED": 1.0, "FUSE": 2.0, "REDUCE": 1.5, "COMBINED": 2.6}
    active, peak = [0], [0]
    lock = threading.Lock()

    def check_factory(min_speedup):
        def check(path):
            with lock:
                active[0] += 1; peak[0] = max(peak[0], active[0])
            time.sleep(0.05)
            tag = open(path).read().split("#")[1].split()[0]
            with lock:
                active[0] -= 1
            sp = speeds[tag]
            return Report(sp >= min_speedup, "ok" if sp >= min_speedup else "speed", speedup=sp,
                          expected=f">= {min_speedup}", saw=f"{sp}x")
        return check

    def factory():
        class P:
            def propose(self, op, progress, feedback, last_code):
                known = " ".join(progress.known)
                if "Combine" in known:
                    tag = "COMBINED"
                elif "Fuse" in known:
                    tag = "FUSE"
                elif "reductions" in known:
                    tag = "REDUCE"
                else:
                    tag = "SEED"
                return f"{tag} approach", f"def forward(x):\n    return x  # {tag} v{id(self)}\n"
        return P()

    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        run, final = run_pipeline("rmsnorm", ["fuse", "reduce"], factory, check_factory, pathlib.Path(d),
                                  max_attempts=2, workers=3, log=lambda s: None)
    assert run.status["merge"] == "done", run.errors
    assert final.winner == "combine" and final.speedup == 2.6
    assert peak[0] == 1                                    # GPU checks never overlapped
    table = {row[0]: row for row in final.table}
    assert table["fuse"][2] and table["reduce"][2]         # both gained over the 1.0x seed
