import executor
from registry import ToolRegistry
from schemas import Plan, Subtask
from working_memory import WorkingMemory

EVIDENCE = "[acme_bakery_policy.md] (distance 1.03)\nRefunds are available within 7 days for unopened packaged goods."
BAD = "Refunds are available within 14 days for unopened goods."  # 14 is not in the evidence
GOOD = "Refunds are available within 7 days for unopened packaged goods."


def make_plan() -> Plan:
    return Plan(
        goal="demo",
        subtasks=[
            Subtask(id=1, description="Research refund policy", specialist="research",
                    expected_output="facts", complexity="low"),
            Subtask(id=2, description="Write summary", specialist="writing",
                    depends_on=[1], expected_output="summary", complexity="low"),
        ],
        confidence=0.9,
    )


def run_with(monkeypatch, fake):
    monkeypatch.setattr(executor, "run_specialist", fake)
    return executor.execute_plan(make_plan(), ToolRegistry())


def test_bad_answer_is_revised_then_approved(monkeypatch):
    calls = []

    def fake(specialist, subtask, registry, context="", feedback=""):
        calls.append((specialist, feedback))
        if specialist == "research":
            return {"success": True, "output": GOOD if feedback else BAD, "steps": 2, "evidence": EVIDENCE}
        return {"success": True, "output": "Summary: refunds are available within 7 days for unopened goods.",
                "steps": 1, "evidence": ""}

    results = run_with(monkeypatch, fake)
    research_calls = [c for c in calls if c[0] == "research"]

    assert results[1]["status"] == "done" and results[2]["status"] == "done"
    assert len(research_calls) == 2
    assert "14" in research_calls[1][1]  # the second attempt received the reviewer's feedback


def test_persistent_rejection_skips_dependents(monkeypatch):
    def fake(specialist, subtask, registry, context="", feedback=""):
        return {"success": True, "output": BAD, "steps": 2, "evidence": EVIDENCE}

    results = run_with(monkeypatch, fake)
    assert results[1]["status"] == "rejected"
    assert results[2]["status"] == "skipped"


def test_crash_is_reported_as_failed_and_skips_dependents(monkeypatch):
    def fake(specialist, subtask, registry, context="", feedback=""):
        raise RuntimeError("402 quota exhausted")

    results = run_with(monkeypatch, fake)
    assert results[1]["status"] == "failed"
    assert results[2]["status"] == "skipped"


def test_executor_records_outputs_in_working_memory(monkeypatch):
    def fake(specialist, subtask, registry, context="", feedback=""):
        out = GOOD if specialist == "research" else "Summary: refunds are available within 7 days for unopened goods."
        return {"success": True, "output": out, "steps": 1, "evidence": EVIDENCE}

    memory = WorkingMemory("demo")
    monkeypatch.setattr(executor, "run_specialist", fake)
    executor.execute_plan(make_plan(), ToolRegistry(), memory)

    assert memory.step_output(1) == GOOD
    assert memory.get("step:2")["status"] == "done"
    assert memory.get("plan")["goal"] == "demo"
    assert memory.errors == []


def test_executor_logs_errors_in_working_memory(monkeypatch):
    def fake(specialist, subtask, registry, context="", feedback=""):
        return {"success": True, "output": BAD, "steps": 1, "evidence": EVIDENCE}

    memory = WorkingMemory("demo")
    monkeypatch.setattr(executor, "run_specialist", fake)
    executor.execute_plan(make_plan(), ToolRegistry(), memory)

    assert [e["step"] for e in memory.errors] == [1, 2]  # step 1 rejected, step 2 skipped