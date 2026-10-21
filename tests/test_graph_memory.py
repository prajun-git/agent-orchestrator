import graph
from long_term_memory import LongTermMemory
from planner import format_memories
from schemas import Plan, Subtask
from test_long_term_memory import FakeEmbedder


def make_state(**over) -> dict:
    base = {
        "user_id": "u1", "task": "", "memories": [], "clarifications": [], "plan": None,
        "rounds": 0, "step_results": {}, "tool_log": [], "result": "",
    }
    base.update(over)
    return base


def two_step_plan() -> Plan:
    return Plan(
        goal="demo",
        subtasks=[
            Subtask(id=1, description="Research", specialist="research", expected_output="x", complexity="low"),
            Subtask(id=2, description="Write", specialist="writing", depends_on=[1],
                    expected_output="x", complexity="low"),
        ],
        confidence=0.9,
    )


def use_fake_memory(monkeypatch) -> LongTermMemory:
    mem = LongTermMemory(FakeEmbedder(), path=None)
    monkeypatch.setattr(graph, "get_memory", lambda: mem)
    return mem


def test_format_memories_empty_and_filled():
    assert format_memories([]) == ""
    text = format_memories([{
        "task": "summarize refund policy", "outcome": "failed",
        "plan": ["research: refunds"], "lessons": ["step 2 ended as skipped"],
    }])
    assert "summarize refund policy" in text and "failed" in text
    assert "research: refunds" in text and "step 2 ended as skipped" in text


def test_run_is_remembered_and_recalled(monkeypatch):
    use_fake_memory(monkeypatch)
    results = {1: {"status": "done"}, 2: {"status": "done"}}
    graph.remember_node(make_state(task="summarize refund policy of acme bakery",
                                   plan=two_step_plan(), step_results=results))

    out = graph.recall_node(make_state(task="summarize the refund policy of acme bakery shop"))
    assert len(out["memories"]) == 1
    assert out["memories"][0]["outcome"] == "success"


def test_gave_up_run_is_stored_as_failed_with_a_lesson(monkeypatch):
    mem = use_fake_memory(monkeypatch)
    vague = Plan(goal="x", subtasks=[], blocking_questions=["What does better mean?"], confidence=0.0)
    graph.remember_node(make_state(task="make my business better", plan=vague))

    found = mem.recall("u1", "make my business better")
    assert found[0]["outcome"] == "failed"
    assert "What does better mean?" in found[0]["lessons"][0]


def test_recall_node_is_per_user(monkeypatch):
    use_fake_memory(monkeypatch)
    graph.remember_node(make_state(task="summarize refund policy of acme bakery", plan=two_step_plan(),
                                   step_results={1: {"status": "done"}, 2: {"status": "done"}}))
    out = graph.recall_node(make_state(user_id="someone_else", task="summarize refund policy of acme bakery"))
    assert out["memories"] == []