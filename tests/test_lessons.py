import pathlib
import tempfile

import pytest

from kernelladder import bench
from kernelladder.lessons import Store, signature
from kernelladder.pipeline import Cand, run_pipeline
from kernelladder.verify import Report

SPEEDS = {"SEED": 1.0, "FUSE": 2.0, "REDUCE": 1.0, "LAUNCH": 1.5}     # only 'fuse' truly helps


def tag_for(known: str) -> str:
    return ("FUSE" if "Fuse" in known else "REDUCE" if "reductions" in known
            else "LAUNCH" if "launch parameters" in known else "SEED")


def fake_run_one(op, lessons):
    def check_factory(min_speedup):
        def check(path):
            sp = SPEEDS[open(path).read().split("#")[1].split()[0]]
            return Report(sp >= min_speedup, "ok" if sp >= min_speedup else "speed", speedup=sp, saw=f"{sp}x")
        return check

    def factory():
        class P:
            def propose(self, o, progress, feedback, last_code):
                t = tag_for(" ".join(progress.known))
                return f"{t} idea", f"def forward(x):\n    return x  # {t} v{id(self)}\n"
        return P()

    with tempfile.TemporaryDirectory() as d:
        return run_pipeline(op, ["fuse", "reduce", "launch"], factory, check_factory, pathlib.Path(d),
                            max_attempts=2, workers=3, log=lambda s: None, lessons=lessons)


def cand(name, gained, base=1.0, speed=1.0, attempts=2, trail=()):
    return Cand(name, "c", speed, gained, "x", attempts, (name,), base, trail)


# ---- store ---------------------------------------------------------------------------------

def test_strategy_lessons_are_graded_not_absolute(tmp_path):
    s = Store(tmp_path / "l.json")
    v0 = s.view("sm_121")
    assert v0.attempts_for("kv_quant", "launch", 2) == 2                       # nothing known
    s.learn("rmsnorm", "sm_121", [cand("launch", False)], "r1")
    assert s.view("sm_121").attempts_for("kv_quant", "launch", 2) == 1         # one miss: probe once
    s.learn("softmax", "sm_121", [cand("launch", False)], "r2")
    assert s.view("sm_121").attempts_for("kv_quant", "launch", 2) == 0         # two misses: skip
    s.learn("attention", "sm_121", [cand("launch", True, speed=1.2)], "r3")
    v = s.view("sm_121")
    assert v.attempts_for("kv_quant", "launch", 2) == 0                        # other family does not transfer
    assert v.attempts_for("attention", "launch", 2) == 2                       # gained there: full budget


def test_lessons_are_filed_per_architecture(tmp_path):
    s = Store(tmp_path / "l.json")
    s.learn("rmsnorm", "sm_121", [cand("launch", False)], "r1")
    assert s.view("sm_80").attempts_for("rmsnorm", "launch", 2) == 2


def test_a_run_can_be_reverted_and_totals_recompute(tmp_path):
    s = Store(tmp_path / "l.json")
    s.learn("rmsnorm", "sm_121", [cand("launch", False)], "bad-run")
    s.learn("softmax", "sm_121", [cand("launch", True, speed=1.3)], "good-run")
    assert s.drop_run("bad-run") == 1
    s2 = Store(tmp_path / "l.json")                                            # persisted
    assert s2.view("sm_121").attempts_for("kv_quant", "launch", 2) == 2
    assert "bad-run" not in s2.describe() and "good-run" in s2.describe()


def test_skipped_strategies_teach_nothing(tmp_path):
    s = Store(tmp_path / "l.json")
    assert s.learn("rmsnorm", "sm_121", [cand("launch", False, attempts=0)], "r1") == 0


def test_recurring_failure_becomes_a_hint_only_after_it_recurs(tmp_path):
    s = Store(tmp_path / "l.json")
    t1 = (("correctness", "output 0: max abs err 1.2e-03 at flat index 55: got 0.1", "plain dot"),
          ("ok", "green", "cast q to k dtype before tl.dot"))
    t2 = (("correctness", "output 0: max abs err 7.9e-02 at flat index 9: got 0.4", "other dot"),
          ("ok", "green", "cast p to v dtype"))
    s.learn("rmsnorm", "sm_121", [cand("seed", True, trail=t1)], "r1")
    assert not [h for h in s.view("sm_121").hints("kv_quant") if "recurring" in h]      # once is an anecdote
    s.learn("softmax", "sm_121", [cand("seed", True, trail=t2)], "r2")
    hints = [h for h in s.view("sm_121").hints("kv_quant") if "recurring" in h]
    assert len(hints) == 1 and "seen 2x" in hints[0] and "cast p to v dtype" in hints[0]


def test_signature_masks_numbers():
    assert signature("max abs err 1.2e-03 at flat index 55") == signature("max abs err 9.9e-01 at flat index 7")


# ---- the method, end to end with a fake model ----------------------------------------------

def test_learning_cuts_search_effort_without_changing_the_result(tmp_path):
    rows = bench.run_bench(["rmsnorm"], ["kv_quant"], "sm_121", tmp_path / "l.json", fake_run_one,
                           gpu_seconds=lambda: 0.0, log=lambda s: None)
    by_arm = {r.arm: r for r in rows}
    assert by_arm["control"].speedup == by_arm["treatment"].speedup == 2.0
    assert by_arm["treatment"].model_attempts < by_arm["control"].model_attempts       # 4 < 6
    assert by_arm["train"].model_attempts == by_arm["control"].model_attempts          # same setup, no lessons


def test_two_misses_skip_the_strategy_outright(tmp_path):
    rows = bench.run_bench(["rmsnorm", "softmax"], ["kv_quant"], "sm_121", tmp_path / "l.json", fake_run_one,
                           gpu_seconds=lambda: 0.0, log=lambda s: None)
    t = next(r for r in rows if r.arm == "treatment")
    assert t.skipped >= 1 and t.speedup == 2.0


def test_report_states_its_own_limits(tmp_path):
    rows = bench.run_bench(["rmsnorm"], ["kv_quant"], "sm_121", tmp_path / "l.json", fake_run_one,
                           gpu_seconds=lambda: 0.0, log=lambda s: None)
    text = bench.report(rows, "sm_121", ["rmsnorm"], ["kv_quant"])
    assert "kv_quant" in text and "model attempts" in text and "**one run**" in text
