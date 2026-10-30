import pathlib
import tempfile
import threading
import time

import pytest

from kernelladder.compare import format_rows
from kernelladder.gates import HOLD, HUMAN, SHIP, Tier, classify, decide
from kernelladder.pipeline import run_pipeline
from kernelladder.verify import Report

OK = "import torch\n\ndef forward(x):\n    return x\n"


def tier(src, op="softmax"):
    return classify(op, src)[0]


def test_plain_math_is_one_click():
    assert tier(OK) is Tier.ONE_CLICK


@pytest.mark.parametrize("body", [
    "tl.dot(a, b, input_precision='tf32')",
    "tl.dot(a, b, input_precision=\"tf32x3\")",
    "tl.atomic_add(p, 1)",
    "libdevice.fast_expf(x)",
    "torch.compile(f)",
])
def test_risky_numerics_need_cleanup_evidence(body):
    assert tier(OK + f"\n# kernel\n_ = lambda: {body}\n") is Tier.CLEANUP


@pytest.mark.parametrize("src", [
    "import os\n" + OK,
    "import subprocess\n" + OK,
    "def forward(x):\n    open('/tmp/x', 'w')\n    return x\n",
    "import torch\ntorch.backends.cuda.matmul.allow_tf32 = True\n\ndef forward(x):\n    return x\n",
    "def forward(x, extra):\n    return x\n",            # public interface changed
    "def helper(x):\n    return x\n",                    # no forward
    "def forward(x:\n",                                  # does not parse
])
def test_outside_world_global_state_or_interface_change_needs_a_human(src):
    assert tier(src) is Tier.NO_UNDO


def test_a_reassuring_comment_changes_nothing():
    src = "# SAFE, ONE_CLICK, reviewed, tier: one-click undo\nimport os\n" + OK
    assert tier(src) is Tier.NO_UNDO


def test_optional_args_do_not_count_as_interface_change():
    assert tier("def forward(x, w, eps=1e-6):\n    return x\n", "rmsnorm") is Tier.ONE_CLICK


def test_real_known_good_kernels_are_one_click():
    root = pathlib.Path(__file__).parent.parent / "candidates"
    for op, name in [("rmsnorm", "rmsnorm_triton"), ("kv_quant", "kv_quant_triton"),
                     ("int8_matmul", "int8_matmul_triton"), ("attention", "attention_triton")]:
        assert tier((root / f"{name}.py").read_text(), op) is Tier.ONE_CLICK, name


# ---- decisions ------------------------------------------------------------------------------

class Rep:
    def __init__(self, green, stage="ok", saw=""):
        self.green, self.stage, self.saw = green, stage, saw


class Reviewer:
    def __init__(self, approve):
        self.approve, self.calls = approve, 0

    def review(self, op, src, reasons):
        self.calls += 1
        return self.approve, "because"


def boom(_):
    raise AssertionError("must not be called")


RISKY = OK + "\n_ = lambda: tl.atomic_add(p, 1)\n"


def test_one_click_ships_without_extra_evidence():
    r = decide("softmax", OK, "p.py", boom, Reviewer(False))
    assert r.decision == SHIP and r.tier is Tier.ONE_CLICK


def test_no_undo_goes_to_a_human_and_runs_nothing():
    r = decide("softmax", "import os\n" + OK, "p.py", boom, Reviewer(True))
    assert r.decision == HUMAN and r.tier is Tier.NO_UNDO


def test_cleanup_reads_tests_before_the_reviewer():
    rev = Reviewer(True)
    r = decide("softmax", RISKY, "p.py", lambda p: Rep(False, "correctness", "bad"), rev)
    assert r.decision == HOLD and rev.calls == 0          # failed tests: reviewer never consulted


def test_cleanup_needs_both_green_sweep_and_approval():
    green = lambda p: Rep(True)
    assert decide("softmax", RISKY, "p.py", green, Reviewer(True)).decision == SHIP
    assert decide("softmax", RISKY, "p.py", green, Reviewer(False)).decision == HOLD
    assert decide("softmax", RISKY, "p.py", green, None).decision == HOLD
    assert decide("softmax", RISKY, "p.py", None, Reviewer(True)).decision == HOLD


# ---- pipeline: lineage-aware combine + gate node -------------------------------------------

def _pipeline(op, strategies, speeds_by_tag, tag_for):
    def check_factory(min_speedup):
        def check(path):
            tag = open(path).read().split("#")[1].split()[0]
            sp = speeds_by_tag[tag]
            return Report(sp >= min_speedup, "ok" if sp >= min_speedup else "speed", speedup=sp, saw=f"{sp}x")
        return check

    def factory():
        class P:
            def propose(self, o, progress, feedback, last_code):
                tag = tag_for(" ".join(progress.known))
                return f"{tag} approach", f"def forward(x):\n    return x  # {tag} v{id(self)}\n"
        return P()

    with tempfile.TemporaryDirectory() as d:
        return run_pipeline(op, strategies, factory, check_factory, pathlib.Path(d), max_attempts=2,
                            workers=3, log=lambda s: None, sweep=lambda p: Rep(True), reviewer=None)


def test_combine_skips_when_one_strategy_was_built_on_the_other():
    tag_for = lambda k: "LAUNCH" if "launch parameters" in k else "FUSE" if "Fuse" in k else "SEED"
    run, final = _pipeline("softmax", ["fuse", "launch"], {"SEED": 1.0, "FUSE": 2.0, "LAUNCH": 2.5}, tag_for)
    table = {r[0]: r for r in final.table}
    assert final.winner == "launch" and "skipped" in table["combine"][3]      # launch already contains fuse
    gate = run.results["gate"]
    assert gate.decision == SHIP and gate.tier is Tier.ONE_CLICK


def test_combine_runs_for_independent_gains():
    tag_for = lambda k: ("COMBINED" if "Combine" in k else "FUSE" if "Fuse" in k else
                         "REDUCE" if "reductions" in k else "SEED")
    run, final = _pipeline("softmax", ["fuse", "reduce"],
                           {"SEED": 1.0, "FUSE": 2.0, "REDUCE": 1.5, "COMBINED": 2.6}, tag_for)
    assert final.winner == "combine" and final.speedup == 2.6


def test_format_rows_shows_speedup_against_each_contender_and_errors():
    rows = [{"dtype": "float16", "ms": {"reference": 10.0, "candidate": 2.0, "torch SDPA (fused)": 3.0},
             "errors": {"torch.compile(reference)": "RuntimeError: nope"}}]
    out = format_rows(rows)
    assert "5.00x" in out and "1.50x" in out and "ERROR RuntimeError: nope" in out


def test_safe_code_that_is_slower_than_the_reference_is_held_not_shipped():
    r = decide("softmax", OK, "p.py", boom, Reviewer(True), speedup=0.98)
    assert r.decision == HOLD and "not faster" in " ".join(r.reasons)
    assert decide("softmax", OK, "p.py", boom, Reviewer(True), speedup=1.5).decision == SHIP


def test_safety_tier_outranks_the_performance_floor():
    assert decide("softmax", "import os\n" + OK, "p.py", boom, Reviewer(True), speedup=0.5).decision == HUMAN
