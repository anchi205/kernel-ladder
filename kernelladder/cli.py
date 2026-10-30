import argparse
from pathlib import Path

from .runner import run_local, run_remote


def _check(a):
    run = run_remote if a.remote else run_local
    return lambda path: run(a.op, path, min_speedup=a.target if hasattr(a, "target") else a.min_speedup)


def cmd_check(a):
    r = _check(a)(a.candidate)
    print(r.feedback())
    raise SystemExit(0 if r.green else 1)


def cmd_optimize(a):
    from .agent import ClaudeCodeProposer, ClaudeProposer
    from .loop import run_loop
    if a.proposer == "api":
        import anthropic
        proposer = ClaudeProposer(anthropic.Anthropic(), model=a.model or "claude-opus-5-5")
    else:
        proposer = ClaudeCodeProposer(model=a.model)
    res = run_loop(a.op, proposer, _check(a), Path(a.out) / a.op, target=a.target, max_attempts=a.max_attempts)
    print(f"\noutcome: {res.outcome}")
    if res.best:
        print(f"best: attempt #{res.best.n}, {res.best.speedup:.2f}x ({res.best.approach}) -> {res.best_path}")
    from .guard import GUARD, alert
    print(f"usage: {proposer.usage}; GPU time on the box: {GUARD.used_s:.0f}s of {GUARD.budget_s:.0f}s budget")
    if res.outcome == "gpu_guard":
        alert("Run stopped early to protect the shared GPU; best result so far is saved.")
    raise SystemExit(0 if res.outcome == "green" else 1)


def _graph_env(a):
    """Proposer factory + check factory + run function, shared by `graph` and `bench`."""
    from .agent import ClaudeCodeProposer, ClaudeProposer
    if a.proposer == "api":
        import anthropic
        client = anthropic.Anthropic()
        factory = lambda: ClaudeProposer(client, model=a.model or "claude-opus-5-5")
    else:
        factory = lambda: ClaudeCodeProposer(model=a.model)
    run_fn = run_remote if a.remote else run_local
    return factory, (lambda op: lambda ms: (lambda path: run_fn(op, path, min_speedup=ms))), run_fn


def _arch(a):
    from .runner import Remote, get_arch
    return get_arch(Remote.from_env() if a.remote else None)


def cmd_graph(a):
    import json
    from dataclasses import asdict
    from datetime import datetime

    from .gates import ClaudeCodeReviewer
    from .guard import GUARD, alert
    from .lessons import Store
    from .pipeline import run_pipeline
    from .strategies import DEFAULT_STRATEGIES, STRATEGIES, plan_text
    names = a.strategies.split(",") if a.strategies else list(DEFAULT_STRATEGIES[a.op])
    unknown = [n for n in names if n not in STRATEGIES]
    if unknown:
        raise SystemExit(f"unknown strategies: {unknown}; choose from {sorted(STRATEGIES)}")
    factory, check_for, run_fn = _graph_env(a)
    store = Store(a.lessons) if a.lessons else None
    arch = _arch(a) if store else None
    view = store.view(arch) if store else None
    if view:
        for line in view.hints(a.op):
            print("using", line)
    print(f"graph for '{a.op}':\n" + plan_text(names) + "\n")
    out = Path(a.out) / a.op / "graph"
    sweep = lambda path: run_fn(a.op, path, min_speedup=0.0, wide=True)
    reviewer = None if a.no_review else ClaudeCodeReviewer(model=a.model)
    run, final = run_pipeline(a.op, names, factory, check_for(a.op), out, max_attempts=a.max_attempts,
                              workers=a.workers, sweep=sweep, reviewer=reviewer, lessons=view,
                              on_event=lambda e: print(f"[{e.t:6.1f}s] {e.node}: {e.kind} {e.detail}"[:200]))
    out.mkdir(parents=True, exist_ok=True)
    (out / "trace.json").write_text(json.dumps([asdict(e) for e in run.events], indent=1))
    print(f"\nwall time {run.wall_s:.0f}s; GPU time on the box {GUARD.used_s:.0f}s of {GUARD.budget_s:.0f}s budget")
    if final is None:
        raise SystemExit(f"graph produced no result; failures: {run.errors}")
    print(f"{'node':<10}{'speedup':>9}  gained  attempts  outcome")
    for name, sp, gained, outcome, att in final.table:
        print(f"{name:<10}{sp:>8.2f}x  {str(gained):<7}{att:>8}  {outcome}")
    conf = f"{final.confirmed_speedup:.2f}x" if final.confirmed_speedup else "n/a"
    print(f"\nWINNER {final.winner}: {final.speedup:.2f}x when produced, {conf} re-measured -> {final.path}")
    gate = run.results.get("gate")
    print("\n" + (gate.text() if gate else f"GATE  did not run ({run.errors.get('gate', 'no result')})"))
    if store and a.learn:
        run_id = f"{a.op}-{datetime.now():%Y%m%d-%H%M%S}"
        n = store.learn(a.op, arch, final.cands, run_id)
        print(f"\nlearned {n} lesson contribution(s) as run '{run_id}' -> {a.lessons}")
    if run.errors:
        print(f"node failures: {run.errors}")
        alert("Graph finished with failed nodes; see the log.")


def cmd_lessons(a):
    from .lessons import Store
    store = Store(a.file)
    if a.drop:
        print(f"removed {store.drop_run(a.drop)} contribution(s) from run '{a.drop}'")
    print(store.describe())


def cmd_bench(a):
    from . import bench
    from .guard import GUARD
    from .pipeline import run_pipeline
    from .strategies import DEFAULT_STRATEGIES
    factory, check_for, _ = _graph_env(a)
    out, arch = Path(a.out), _arch(a)
    train, held = a.train.split(","), a.heldout.split(",")

    def run_one(op, lessons):
        sub = out / ("treatment" if lessons else "control") / op
        return run_pipeline(op, list(DEFAULT_STRATEGIES[op]), factory, check_for(op), sub,
                            max_attempts=a.max_attempts, workers=a.workers, lessons=lessons, log=lambda s: None)

    store_path = out / f"lessons_{arch}.json"
    if store_path.exists():
        store_path.unlink()                        # a benchmark learns from its own train ops only
    rows = bench.run_bench(train, held, arch, store_path, run_one, lambda: GUARD.used_s)
    text = bench.report(rows, arch, train, held)
    bench.save(rows, text, out)
    print("\n" + text + f"\n\nsaved to {out}/report.md")


def cmd_compare(a):
    from .compare import format_rows
    from .runner import run_compare
    print(f"{a.op}: {a.candidate}" + format_rows(run_compare(a.op, a.candidate)))


def cmd_gate(a):
    from .gates import ClaudeCodeReviewer, decide
    run_fn = run_remote if a.remote else run_local
    src = Path(a.candidate).read_text()
    res = decide(a.op, src, a.candidate, lambda path: run_fn(a.op, path, min_speedup=0.0, wide=True),
                 None if a.no_review else ClaudeCodeReviewer())
    print(res.text())
    raise SystemExit({"ship": 0, "hold": 2, "human": 3}[res.decision])


def cmd_usage(a):
    from . import usage
    print(usage.summary(usage.DEFAULT_LOG))


def main():
    ap = argparse.ArgumentParser(prog="kernelladder")
    sub = ap.add_subparsers(dest="cmd", required=True)
    remote_help = "run on the GPU container (KL_SSH_TARGET, KL_CONTAINER, optional KL_SSH_KEY)"

    c = sub.add_parser("check", help="verify one candidate file")
    c.add_argument("op"); c.add_argument("candidate")
    c.add_argument("--min-speedup", type=float, default=1.0)
    c.add_argument("--remote", action="store_true", help=remote_help)
    c.set_defaults(fn=cmd_check)

    o = sub.add_parser("optimize", help="let Claude write and iterate on a kernel")
    o.add_argument("op")
    o.add_argument("--target", type=float, default=1.0, help="required speedup vs the reference")
    o.add_argument("--max-attempts", type=int, default=8)
    o.add_argument("--proposer", choices=["code", "api"], default="code",
                   help="code: local `claude -p` (uses your subscription); api: Anthropic API (needs credentials)")
    o.add_argument("--model", default=None, help="model for the proposer (default: the proposer's own default)")
    o.add_argument("--out", default="runs")
    o.add_argument("--remote", action="store_true", help=remote_help)
    o.set_defaults(fn=cmd_optimize)

    g = sub.add_parser("graph", help="parallel strategy graph: seed, strategies, combine, merge")
    g.add_argument("op")
    g.add_argument("--strategies", help="comma list, default depends on the op")
    g.add_argument("--max-attempts", type=int, default=2, help="attempts per strategy node")
    g.add_argument("--workers", type=int, default=3, help="parallel model calls (GPU checks stay serialized)")
    g.add_argument("--proposer", choices=["code", "api"], default="code")
    g.add_argument("--model", default=None)
    g.add_argument("--out", default="runs")
    g.add_argument("--remote", action="store_true", help=remote_help)
    g.add_argument("--no-review", action="store_true", help="skip the second-opinion reviewer (cleanup-tier code then holds)")
    g.add_argument("--lessons", help="lessons file: apply what it holds to this run")
    g.add_argument("--learn", action="store_true", help="after the run, add what it taught to the --lessons file")
    g.set_defaults(fn=cmd_graph)

    ls = sub.add_parser("lessons", help="show a lessons file, or remove one run's lessons")
    ls.add_argument("file"); ls.add_argument("--drop", metavar="RUN_ID")
    ls.set_defaults(fn=cmd_lessons)

    b = sub.add_parser("bench", help="train on some ops, measure on held-out ops with and without lessons")
    b.add_argument("--train", required=True, help="comma list, e.g. rmsnorm")
    b.add_argument("--heldout", required=True, help="comma list, e.g. kv_quant")
    b.add_argument("--max-attempts", type=int, default=2)
    b.add_argument("--workers", type=int, default=3)
    b.add_argument("--proposer", choices=["code", "api"], default="code")
    b.add_argument("--model", default=None)
    b.add_argument("--out", default="runs/bench")
    b.add_argument("--remote", action="store_true", help=remote_help)
    b.set_defaults(fn=cmd_bench)

    cp = sub.add_parser("compare", help="time a candidate against the reference and library baselines (GPU)")
    cp.add_argument("op"); cp.add_argument("candidate")
    cp.set_defaults(fn=cmd_compare)

    us = sub.add_parser("usage", help="how much shared-GPU time KernelLadder has held, across all sessions")
    us.set_defaults(fn=cmd_usage)

    gt = sub.add_parser("gate", help="classify a candidate by undo cost and decide ship / hold / human")
    gt.add_argument("op"); gt.add_argument("candidate")
    gt.add_argument("--remote", action="store_true", help=remote_help)
    gt.add_argument("--no-review", action="store_true")
    gt.set_defaults(fn=cmd_gate)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
