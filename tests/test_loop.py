from pathlib import Path

from kernelladder.agent import ScriptedProposer, parse_reply
from kernelladder.loop import run_loop
from kernelladder.runner import run_local

C = Path(__file__).parent.parent / "candidates"
src = lambda name: (C / name).read_text()
check = lambda path: run_local("softmax", path, min_speedup=0.0)


def test_green_after_a_fix(tmp_path):
    p = ScriptedProposer([("no max-subtraction", src("softmax_naive_unstable.py")),
                          ("subtract row max", src("softmax_stable.py"))])
    res = run_loop("softmax", p, check, tmp_path, target=0.0, log=lambda s: None)
    assert res.outcome == "green" and len(res.progress.attempts) == 2
    assert res.progress.attempts[0].stage == "correctness"
    assert "no max-subtraction" in (tmp_path / "PROGRESS.md").read_text()


def test_same_failure_twice_escalates(tmp_path):
    bad = src("softmax_naive_unstable.py")
    p = ScriptedProposer([("try A", bad), ("try B", bad), ("try C", bad)])
    res = run_loop("softmax", p, check, tmp_path, log=lambda s: None)
    assert res.outcome == "stuck" and len(res.progress.attempts) == 2


def test_speed_plateau_stops(tmp_path):
    slow = src("softmax_stable.py")   # correct but below the target every time
    p = ScriptedProposer([("A", slow), ("B", slow), ("C", slow), ("D", slow)])
    res = run_loop("softmax", p, lambda path: run_local("softmax", path, min_speedup=50.0),
                   tmp_path, target=50.0, log=lambda s: None)
    assert res.outcome == "plateau" and res.best_path and res.best_path.exists()


def test_attempt_budget(tmp_path):
    cheat = src("cheat_calls_op.py")
    p = ScriptedProposer([(f"cheat {i}", cheat + f"\n# v{i}\n" * i) for i in range(1, 4)])
    res = run_loop("softmax", p, check, tmp_path, max_attempts=1, log=lambda s: None)
    assert res.outcome == "budget"


def test_parse_reply():
    approach, code = parse_reply("APPROACH: fuse it\n```python\nx = 1\n```")
    assert approach == "fuse it" and code == "x = 1\n"


def test_claude_code_proposer_runs_headless_and_parses(monkeypatch):
    import subprocess

    from kernelladder import agent
    from kernelladder.progress import Progress

    seen = {}

    def fake_run(cmd, input, capture_output, text, timeout, cwd):
        seen.update(cmd=cmd, input=input, cwd=cwd)
        return subprocess.CompletedProcess(cmd, 0, "APPROACH: fuse\n```python\ndef forward(x):\n    return x\n```", "")

    monkeypatch.setattr(agent.subprocess, "run", fake_run)
    pr = agent.ClaudeCodeProposer(model="opus")
    approach, code = pr.propose("rmsnorm", Progress(goal="g"), None, None)
    assert approach == "fuse" and "def forward" in code
    assert seen["cmd"][:2] == ["claude", "-p"] and "--tools" in seen["cmd"] and "opus" in seen["cmd"]
    assert "Reference implementation" in seen["input"] and "PROGRESS.md" in seen["input"]
