"""Proposers: the part that writes code. Swappable; the loop never trusts either."""
import inspect
import re
import subprocess
import tempfile

from .ops import OPS
from .progress import Progress

SYSTEM = """You write fast GPU kernels in PyTorch/Triton, one candidate per turn.

Reply with exactly:
APPROACH: <one line naming the optimization you are trying>
```python
<the complete candidate file>
```

Rules:
- The file must define forward(*inputs) returning a tensor with the same shape and dtype as the reference.
- Implement the op yourself. Do not call the library function that is the op, and do not import kernelladder.
- You may use torch ops, torch.compile, or Triton kernels (import triton, triton.language as tl).
- Never repeat an approach listed under DEAD ENDS. If the feedback shows the same failure twice, change approach.
- Correctness comes first; speed is measured only after correctness passes.
"""


def _task(op_name: str, hardware: str) -> str:
    op = OPS[op_name]
    return (f"Op: {op_name}\nHardware: {hardware}\nBenchmark shape: {op.bench_shape} float32\n"
            f"Correctness shapes: {op.shapes}\nReference implementation:\n```python\n{inspect.getsource(op.reference)}```\n"
            f"Input generator:\n```python\n{inspect.getsource(OPS[op_name].make_inputs)}```")


def parse_reply(text: str) -> tuple[str, str]:
    m = re.search(r"```python\n(.*?)```", text, re.S)
    if not m:
        raise ValueError("reply had no ```python block")
    a = re.search(r"APPROACH:\s*(.+)", text)
    return (a.group(1).strip() if a else "unnamed approach"), m.group(1)


def build_prompt(op, hardware, progress: Progress, feedback, last_code) -> str:
    # Each turn is built from PROGRESS.md, not from chat history: memory lives in the file.
    parts = [_task(op, hardware), "PROGRESS.md:\n" + progress.render()]
    if feedback and last_code:
        parts += ["Your last candidate:\n```python\n" + last_code + "```", "Check result:\n" + feedback]
    return "\n\n".join(parts)


class ClaudeProposer:
    def __init__(self, client, model: str = "claude-opus-5-5", effort: str = "high",
                 hardware: str = "NVIDIA GB10 (sm_121), torch 2.14 + CUDA 13, triton 3.8"):
        self.client, self.model, self.effort, self.hardware = client, model, effort, hardware
        self.usage = {"input": 0, "output": 0}

    def propose(self, op, progress: Progress, feedback, last_code):
        resp = self.client.messages.create(
            model=self.model, max_tokens=16000, system=SYSTEM,
            thinking={"type": "adaptive"}, output_config={"effort": self.effort},
            messages=[{"role": "user", "content": build_prompt(op, self.hardware, progress, feedback, last_code)}],
        )
        self.usage["input"] += resp.usage.input_tokens
        self.usage["output"] += resp.usage.output_tokens
        text = next((b.text for b in resp.content if b.type == "text"), "")
        return parse_reply(text)


class ScriptedProposer:
    """Replays fixed (approach, code) pairs. Used to test the loop without a model."""

    def __init__(self, script: list[tuple[str, str]]):
        self.script, self.i = script, 0

    def propose(self, op, progress, feedback, last_code):
        item = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return item


class ClaudeCodeProposer:
    """Proposes through the local `claude -p` CLI, so it runs on a Claude subscription, not API credit.

    Tools are disabled and the process starts in an empty directory with no settings loaded:
    the model can only reply with text, and cannot touch the project or the reference ops.
    """

    def __init__(self, model: str | None = None, effort: str | None = "high", timeout: int = 600,
                 hardware: str = "NVIDIA GB10 (sm_121), torch 2.14 + CUDA 13, triton 3.8"):
        self.model, self.effort, self.timeout, self.hardware = model, effort, timeout, hardware
        self.usage = {"calls": 0}

    def command(self) -> list[str]:
        cmd = ["claude", "-p", "--tools", "", "--system-prompt", SYSTEM, "--no-session-persistence",
               "--setting-sources", "", "--output-format", "text"]
        if self.model:
            cmd += ["--model", self.model]
        if self.effort:
            cmd += ["--effort", self.effort]
        return cmd

    def propose(self, op, progress: Progress, feedback, last_code):
        prompt = build_prompt(op, self.hardware, progress, feedback, last_code)
        with tempfile.TemporaryDirectory() as empty:
            p = subprocess.run(self.command(), input=prompt, capture_output=True, text=True,
                               timeout=self.timeout, cwd=empty)
        if p.returncode != 0:
            raise RuntimeError(f"claude -p failed ({p.returncode}): {p.stderr.strip()[-300:]}")
        self.usage["calls"] += 1
        return parse_reply(p.stdout)
