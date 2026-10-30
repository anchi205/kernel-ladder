import pytest

from kernelladder.guard import Guard, GpuBudget, GpuBusy


def mk(**kw):
    sent = []
    return Guard(busy_util=25, busy_wait_s=30, budget_s=60, notify=sent.append, sleep=lambda s: None, **kw), sent


def test_idle_gpu_runs_without_alert():
    g, sent = mk()
    g.before(lambda: 0)
    assert sent == []


def test_busy_gpu_alerts_once_then_proceeds_when_it_frees_up():
    g, sent = mk()
    utils = iter([80, 80, 10])
    g.before(lambda: next(utils))
    assert len(sent) == 1 and "busy" in sent[0]


def test_busy_gpu_that_never_frees_up_stops_the_run():
    g, sent = mk()
    with pytest.raises(GpuBusy):
        g.before(lambda: 90)
    assert len(sent) == 1


def test_budget_alert_at_budget_and_hard_stop_at_double():
    g, sent = mk()
    g.after(59); assert sent == []
    g.after(2); assert len(sent) == 1 and "budget" in sent[0]
    g.after(10); assert len(sent) == 1          # alerts once
    g.before(lambda: 0)                         # 71s: over budget but under 2x, still allowed
    g.after(50)
    with pytest.raises(GpuBudget):
        g.before(lambda: 0)


def test_usage_log_persists_and_triggers_one_daily_alert(tmp_path):
    from kernelladder import usage
    sent = []
    log = tmp_path / "u.jsonl"
    g = Guard(busy_util=25, busy_wait_s=30, budget_s=10_000, daily_s=100, log=log,
              notify=sent.append, sleep=lambda s: None)
    g.after(60, "check:softmax"); assert sent == []
    g.after(60, "compare:attention"); assert len(sent) == 1 and "Today's total" in sent[0]
    g.after(60, "check:softmax"); assert len(sent) == 1
    # a second process (fresh Guard) sees the same history: the daily total is cross-session
    g2 = Guard(daily_s=100, budget_s=10_000, log=log, notify=sent.append, sleep=lambda s: None)
    g2.after(1, "check:rmsnorm")
    assert len(sent) == 2
    assert usage.seconds_today(log) == 181
    text = usage.summary(log)
    assert "check" in text and "compare" in text and "4 runs" in text


def test_usage_log_survives_a_corrupt_line(tmp_path):
    from kernelladder import usage
    log = tmp_path / "u.jsonl"
    usage.record(log, 5, "x")
    log.write_text(log.read_text() + "not json\n")
    usage.record(log, 7, "y")
    assert usage.seconds_today(log) == 12
