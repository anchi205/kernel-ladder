from pathlib import Path

from kernelladder.runner import run_local

C = Path(__file__).parent.parent / "candidates"


def test_wrong_candidate_fails_correctness():
    r = run_local("softmax", str(C / "softmax_naive_unstable.py"))
    assert not r.green and r.stage == "correctness"
    assert "EXPECTED" in r.feedback() and "SAW" in r.feedback()


def test_correct_candidate_passes_correctness():
    r = run_local("softmax", str(C / "softmax_stable.py"), min_speedup=0.0)
    assert r.green, r.feedback()


def test_slow_candidate_fails_speed_gate():
    r = run_local("softmax", str(C / "softmax_stable.py"), min_speedup=100.0)
    assert not r.green and r.stage == "speed"


def test_missing_forward_fails_load(tmp_path):
    p = tmp_path / "bad.py"
    p.write_text("x = 1\n")
    r = run_local("softmax", str(p))
    assert not r.green and r.stage == "load"


import os

import pytest

from kernelladder.runner import run_remote

remote = pytest.mark.skipif("KL_SSH_TARGET" not in os.environ, reason="set KL_SSH_TARGET and KL_CONTAINER")


@remote
def test_remote_wrong_candidate_fails():
    r = run_remote("softmax", str(C / "softmax_naive_unstable.py"))
    assert not r.green and r.stage == "correctness"


@remote
def test_remote_correct_candidate_passes():
    r = run_remote("softmax", str(C / "softmax_stable.py"), min_speedup=0.0)
    assert r.green, r.feedback()


@remote
def test_remote_missing_forward_fails_load(tmp_path):
    p = tmp_path / "bad.py"
    p.write_text("x = 1\n")
    r = run_remote("softmax", str(p))
    assert not r.green and r.stage == "load"


@remote
def test_remote_gpu_hog_is_capped():
    r = run_remote("softmax", str(C / "hog_gpu.py"))
    assert not r.green and "OutOfMemory" in r.saw, r.feedback()


@remote
def test_remote_ram_hog_is_killed(monkeypatch):
    monkeypatch.setenv("KL_MAX_RAM_GB", "4")
    r = run_remote("softmax", str(C / "hog_ram.py"))
    assert not r.green and "exceeded cap" in r.saw, r.feedback()


def test_calling_the_op_itself_is_fenced():
    r = run_local("softmax", str(C / "cheat_calls_op.py"), min_speedup=0.0)
    assert not r.green and r.stage == "fence"


@remote
def test_remote_nondeterministic_candidate_is_caught():
    r = run_remote("softmax", str(C / "softmax_racy.py"), min_speedup=0.0)
    assert not r.green and r.stage == "determinism", r.feedback()


@remote
def test_remote_green_report_carries_profile():
    r = run_remote("rmsnorm", str(C / "rmsnorm_triton.py"))
    assert r.green, r.feedback()
    text = r.feedback()
    assert "PROFILE    candidate" in text and "PROFILE    reference" in text and "% of peak" in text


@remote
@pytest.mark.parametrize("op,cand", [("attention", "attention_triton"), ("int8_matmul", "int8_matmul_triton"),
                                     ("kv_quant", "kv_quant_triton")])
def test_remote_known_good_kernels_pass(op, cand):
    r = run_remote(op, str(C / f"{cand}.py"), min_speedup=0.0)
    assert r.green, r.feedback()
    assert "PROFILE    candidate" in r.feedback()


@remote
@pytest.mark.parametrize("op,cand,needle", [
    ("attention", "attention_noncausal", "max abs err"),
    ("int8_matmul", "int8_matmul_noscale", "max abs err"),
    ("kv_quant", "kv_quant_mutates", "changed in place"),
    ("kv_quant", "kv_quant_one_output", "1 output(s), reference returns 2"),
])
def test_remote_wrong_kernels_fail_correctness(op, cand, needle):
    r = run_remote(op, str(C / f"{cand}.py"), min_speedup=0.0)
    assert not r.green and r.stage == "correctness" and needle in r.saw, r.feedback()
