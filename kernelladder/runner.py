"""Harness wall: candidates run in a subprocess with a hard timeout, never in-process.

Two runners share one contract (op name + candidate path in, Report out):
  run_local   subprocess on this machine
  run_remote  subprocess inside a docker container on a GPU box, over ssh
"""
import functools
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

from .guard import GUARD
from .verify import Report

PACKAGE_DIR = Path(__file__).parent


def _parse(p: subprocess.CompletedProcess) -> Report:
    if p.returncode != 0:
        err = p.stderr.strip().splitlines()
        return Report(False, "load", "worker exits cleanly", err[-1] if err else "crash")
    return Report(**json.loads(p.stdout.strip().splitlines()[-1]))


def run_local(op: str, candidate: str, timeout: int = 120, min_speedup: float = 1.0, wide: bool = False) -> Report:
    cmd = [sys.executable, "-m", "kernelladder.worker", op, candidate, str(min_speedup)] + (["wide"] if wide else [])
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return Report(False, "correctness", f"finish within {timeout}s", "timeout")
    return _parse(p)


@dataclass(frozen=True)
class Remote:
    target: str                     # user@host
    container: str
    key: str | None = None
    python: str = "~/kl-venv/bin/python"
    project_dir: str = "~/work/kernelladder"

    @classmethod
    def from_env(cls) -> "Remote":
        """KL_SSH_TARGET and KL_CONTAINER are required; the rest have defaults."""
        try:
            target, container = os.environ["KL_SSH_TARGET"], os.environ["KL_CONTAINER"]
        except KeyError as e:
            raise SystemExit(f"remote runner needs {e.args[0]} set (see README)")
        return cls(target, container, os.environ.get("KL_SSH_KEY"),
                   os.environ.get("KL_REMOTE_PYTHON", cls.python),
                   os.environ.get("KL_REMOTE_DIR", cls.project_dir))

    def _ssh(self) -> list[str]:
        ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
        if self.key:
            ssh += ["-i", os.path.expanduser(self.key)]
        return ssh

    def gpu_util(self) -> float:
        """Median GPU utilization (%) over ~1s, read on the host so it includes everyone's work."""
        cmd = ("for i in 1 2 3; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; "
               "sleep 0.5; done")
        p = subprocess.run(self._ssh() + [self.target, cmd], capture_output=True, text=True, timeout=30)
        vals = sorted(float(v) for v in p.stdout.split() if v.replace(".", "", 1).isdigit())
        return vals[len(vals) // 2] if vals else 0.0

    def _exec(self, script: str, stdin: bytes, timeout: int) -> subprocess.CompletedProcess:
        ssh = self._ssh()
        inner = f"docker exec -i {shlex.quote(self.container)} bash -lc {shlex.quote(script)}"
        return subprocess.run(ssh + [self.target, inner], input=stdin,
                              capture_output=True, timeout=timeout)


def _text(p: subprocess.CompletedProcess) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(p.args, p.returncode,
                                       p.stdout.decode(errors="replace"), p.stderr.decode(errors="replace"))


@functools.lru_cache(maxsize=None)
def sync_remote(remote: Remote) -> None:
    """Ship the verifier and reference ops so the box never checks against stale code.

    Candidates are not synced here; each run sends its own file on stdin.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.add(PACKAGE_DIR, arcname="kernelladder",
                filter=lambda ti: None if "__pycache__" in ti.name else ti)
    p = _text(remote._exec(f"mkdir -p {remote.project_dir} && tar -xf - -C {remote.project_dir}",
                           buf.getvalue(), timeout=60))
    if p.returncode != 0:
        raise RuntimeError(f"remote sync failed: {p.stderr.strip()[-300:]}")


def _remote_worker(remote: Remote, candidate: str, worker_args: str, timeout: int, tag: str = "") -> subprocess.CompletedProcess:
    """Run kernelladder.worker in the container on one candidate file, behind the shared-GPU guard."""
    sync_remote(remote)
    # KL_MAX_* caps are forwarded; `timeout -s KILL` ends the worker inside the container,
    # since dropping the ssh connection alone would leave a runaway kernel running.
    caps = " ".join(f"{k}={shlex.quote(v)}" for k, v in os.environ.items() if k.startswith("KL_MAX_"))
    script = (f"f=$(mktemp /tmp/kl_XXXXXX.py); cat > $f; "
              f"cd {remote.project_dir} && {caps} timeout -s KILL {max(timeout - 10, 1)} "
              f"{remote.python} -m kernelladder.worker {worker_args.replace('CAND', '$f')}; "
              f"rc=$?; rm -f $f; exit $rc")
    GUARD.before(remote.gpu_util)
    t0 = time.monotonic()
    try:
        return _text(remote._exec(script, Path(candidate).read_bytes(), timeout))
    finally:
        GUARD.after(time.monotonic() - t0, tag)


def run_remote(op: str, candidate: str, remote: Remote | None = None, timeout: int = 180,
               min_speedup: float = 1.0, wide: bool = False) -> Report:
    remote = remote or Remote.from_env()
    args = f"{shlex.quote(op)} CAND {min_speedup}" + (" wide" if wide else "")
    try:
        return _parse(_remote_worker(remote, candidate, args, timeout, tag=f"check:{op}"))
    except subprocess.TimeoutExpired:
        return Report(False, "correctness", f"finish within {timeout}s", "timeout")


def run_compare(op: str, candidate: str, remote: Remote | None = None, timeout: int = 600) -> list[dict]:
    """Time the candidate against the reference and the library baselines (needs the GPU)."""
    if remote is None and "KL_SSH_TARGET" in os.environ:
        remote = Remote.from_env()
    if remote is None:
        cmd = [sys.executable, "-m", "kernelladder.worker", "compare", op, candidate]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    else:
        p = _remote_worker(remote, candidate, f"compare {shlex.quote(op)} CAND", timeout, tag=f"compare:{op}")
    if p.returncode != 0:
        raise RuntimeError(f"compare failed: {p.stderr.strip()[-400:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


@functools.lru_cache(maxsize=None)
def get_arch(remote: Remote | None = None) -> str:
    """GPU architecture tag such as 'sm_121' (or 'cpu'), the key lessons are filed under."""
    code = ("import torch;c=torch.cuda.get_device_capability() if torch.cuda.is_available() else None;"
            "print('sm_%d%d' % c if c else 'cpu')")
    if remote is None:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        return p.stdout.strip().splitlines()[-1] if p.returncode == 0 and p.stdout.strip() else "cpu"
    p = _text(remote._exec(f"cd {remote.project_dir} && {remote.python} -c {shlex.quote(code)}", b"", 120))
    return p.stdout.strip().splitlines()[-1] if p.returncode == 0 and p.stdout.strip() else "unknown"
