"""Gate on the price of undo, decided by code, never by model confidence.

ONE_CLICK  ships on green. Plain math in the candidate's own scope: reverting is deleting a file.
CLEANUP    green is not enough. Reduced precision, atomics, approximate math or JIT make a wrong
           answer subtle and environment-dependent, so it needs a wider test sweep AND an
           independent reviewer (separate context, separate prompt). Evidence is read in ladder
           order: tests first, then the second opinion; the proposer's own confidence is never read.
NO_UNDO    a person signs. The candidate touches the outside world (files, processes, network),
           changes global library state, or changes the op's public interface.

The classifier only reads source text. The model proposes code; this file decides what it unlocks.
"""
import ast
import inspect
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .ops import OPS

SHIP, HOLD, HUMAN = "ship", "hold", "human"


class Tier(str, Enum):
    ONE_CLICK = "one-click undo"
    CLEANUP = "cleanup undo"
    NO_UNDO = "no undo"


FORBIDDEN_IMPORTS = {"os", "subprocess", "socket", "sys", "ctypes", "pathlib", "shutil", "urllib", "requests",
                     "http", "pickle", "importlib", "builtins", "tempfile", "multiprocessing", "threading",
                     "signal", "asyncio"}
FORBIDDEN_CALLS = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint"}

# (pattern, tier it forces, reason). NO_UNDO patterns change process-wide state.
PATTERNS = [
    (r"torch\.backends\.|torch\.set_|torch\.use_deterministic|set_float32_matmul_precision|os\.environ",
     Tier.NO_UNDO, "changes global torch/process state that outlives the kernel"),
    (r"input_precision\s*=\s*['\"](tf32|tf32x3|bf16x3)['\"]|allow_tf32\s*=\s*True",
     Tier.CLEANUP, "reduced-precision matmul (TF32-class): correct on tested inputs, drifts on others"),
    (r"atomic_(add|max|min|cas|xchg|and|or|xor)", Tier.CLEANUP, "atomics: result order can vary run to run"),
    (r"libdevice|fast_math|fastmath|\.approx|rcp\.|ex2\.approx", Tier.CLEANUP, "approximate math"),
    (r"torch\.compile|torch\.jit|torch\.cuda\.graph|CUDAGraph", Tier.CLEANUP,
     "JIT / graph capture: behavior depends on the environment and first-call state"),
    (r"inline_asm|asm_volatile|tl\.inline_asm_elementwise", Tier.CLEANUP, "inline assembly"),
]


@dataclass(frozen=True)
class GateResult:
    tier: Tier
    decision: str                       # ship | hold | human
    reasons: tuple
    evidence: dict = field(default_factory=dict)

    def text(self) -> str:
        head = {SHIP: "SHIP", HOLD: "HOLD", HUMAN: "NEEDS A HUMAN"}[self.decision]
        lines = [f"GATE  {head}   tier: {self.tier.value}"] + [f"  - {r}" for r in self.reasons]
        lines += [f"  evidence {k}: {v}" for k, v in self.evidence.items()]
        return "\n".join(lines)


def classify(op_name: str, source: str) -> tuple[Tier, list[str]]:
    reasons: dict[Tier, list[str]] = {Tier.NO_UNDO: [], Tier.CLEANUP: []}
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return Tier.NO_UNDO, [f"does not parse: {e.msg}"]

    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods = [node.module.split(".")[0]]
        for m in mods:
            if m in FORBIDDEN_IMPORTS:
                reasons[Tier.NO_UNDO].append(f"imports '{m}' (reaches outside the kernel's own scope)")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            reasons[Tier.NO_UNDO].append(f"calls {node.func.id}()")

    for pattern, tier, why in PATTERNS:
        if re.search(pattern, source):
            reasons[tier].append(why)

    fwd = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "forward"), None)
    if fwd is None:
        reasons[Tier.NO_UNDO].append("no forward() at module level")
    else:
        ref = inspect.signature(OPS[op_name].reference)
        need_ref = sum(1 for p in ref.parameters.values() if p.default is p.empty)
        need_cand = len(fwd.args.args) - len(fwd.args.defaults)
        if need_cand != need_ref or len(fwd.args.args) < need_ref:
            reasons[Tier.NO_UNDO].append(
                f"public interface changed: forward takes {need_cand} required argument(s), reference takes {need_ref}")

    for tier in (Tier.NO_UNDO, Tier.CLEANUP):
        if reasons[tier]:
            return tier, sorted(set(reasons[tier]))
    return Tier.ONE_CLICK, ["plain math in its own scope; revert is deleting the file"]


REVIEWER_SYSTEM = """You are an independent code reviewer for GPU kernels. You did not write this code and you have
not seen the author's reasoning. Judge only the source you are given.

Check: does it implement the stated op correctly for all shapes and dtypes (not only common ones)? Are the flagged
risks acceptable here (precision, atomics, approximate math, JIT)? Any out-of-bounds access, race, hidden state,
or behavior that depends on the machine?

Reply with exactly one line first: `VERDICT: APPROVE` or `VERDICT: REJECT`, then at most 5 lines of reasons."""


class ClaudeCodeReviewer:
    """Second opinion through `claude -p`: tools off, empty directory, its own system prompt."""

    def __init__(self, model: str | None = None, timeout: int = 300):
        self.model, self.timeout = model, timeout

    def review(self, op_name: str, source: str, reasons: list[str]) -> tuple[bool, str]:
        import inspect as _i
        prompt = (f"Op: {op_name}\nReference implementation:\n```python\n{_i.getsource(OPS[op_name].reference)}```\n"
                  f"Automatic risk flags on the candidate:\n" + "\n".join(f"- {r}" for r in reasons) +
                  f"\n\nCandidate source:\n```python\n{source}\n```")
        cmd = ["claude", "-p", "--tools", "", "--system-prompt", REVIEWER_SYSTEM, "--no-session-persistence",
               "--setting-sources", "", "--output-format", "text"]
        if self.model:
            cmd += ["--model", self.model]
        with tempfile.TemporaryDirectory() as empty:
            p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=self.timeout, cwd=empty)
        if p.returncode != 0:
            return False, f"reviewer failed: {p.stderr.strip()[-200:]}"
        m = re.search(r"VERDICT:\s*(APPROVE|REJECT)", p.stdout)
        return (m is not None and m.group(1) == "APPROVE"), p.stdout.strip().replace("\n", " ")[:300]


def decide(op_name: str, source: str, path: str, sweep: Callable[[str], object] | None,
           reviewer=None, speedup: float | None = None) -> GateResult:
    """Safety first (undo cost), then a performance floor: safe code that is not faster than the reference has nothing to ship."""
    tier, reasons = classify(op_name, source)
    if tier is Tier.NO_UNDO:
        return GateResult(tier, HUMAN, tuple(reasons))
    if speedup is not None and speedup < 1.0:
        return GateResult(tier, HOLD, tuple(reasons) + (f"not faster than the reference ({speedup:.2f}x): nothing to ship",),
                          {"speedup": f"{speedup:.2f}x"})
    if tier is Tier.ONE_CLICK:
        return GateResult(tier, SHIP, tuple(reasons), {"rule": "ships on green"})

    evidence = {}
    if sweep is None:
        return GateResult(tier, HOLD, tuple(reasons), {"wide sweep": "not available"})
    rep = sweep(path)                               # 1st rung: tests, wider than the normal check
    evidence["wide sweep"] = "green" if rep.green else f"FAILED at {rep.stage}: {rep.saw[:120]}"
    if not rep.green:
        return GateResult(tier, HOLD, tuple(reasons), evidence)
    if reviewer is None:
        evidence["second opinion"] = "no reviewer configured"
        return GateResult(tier, HOLD, tuple(reasons), evidence)
    approved, why = reviewer.review(op_name, source, reasons)   # 2nd rung: independent reviewer
    evidence["second opinion"] = ("APPROVE: " if approved else "REJECT: ") + why
    return GateResult(tier, SHIP if approved else HOLD, tuple(reasons), evidence)
