from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

import graph
from schemas import Plan, Subtask

PROBLEM = {
    1: {"status": "no_data", "output": "The knowledge base has no information on this topic."},
    2: {"status": "skipped", "output": "skipped: step(s) [1] did not finish"},
}
CLEAN = {1: {"status": "done", "output": "ok"}, 2: {"status": "done", "output": "ok"}}


def make_plan() -> Plan:
    return Plan(
        goal="demo",
        subtasks=[
            Subtask(id=1, description="Research", specialist="research", expected_output="x", complexity="low"),
            Subtask(id=2, description="Write", specialist="writing", depends_on=[1],
                    expected_output="x", complexity="low"),
        ],
        confidence=0.9,
    )


def make_state(results, **over) -> dict:
    base = {
        "user_id": "u1", "task": "demo task", "memories": [], "clarifications": [], "plan": make_plan(),
        "rounds": 0, "step_results": results, "tool_log": [], "result": "partial result", "decision": "",
    }
    base.update(over)
    return base


def answer_with(monkeypatch, answer):
    seen = []

    def fake_interrupt(payload):
        seen.append(payload)
        return answer

    monkeypatch.setattr(graph, "interrupt", fake_interrupt)
    return seen


def test_clean_run_does_not_pause(monkeypatch):
    def must_not_pause(payload):
        raise AssertionError("interrupt() must not be called when nothing went wrong")

    monkeypatch.setattr(graph, "interrupt", must_not_pause)
    out = graph.escalation_node(make_state(CLEAN))
    assert out == {"decision": "continue"}


def test_problem_pauses_with_context_and_ignores_skipped_steps(monkeypatch):
    seen = answer_with(monkeypatch, {"action": "accept"})
    out = graph.escalation_node(make_state(PROBLEM))

    assert len(seen) == 1
    assert [p["step"] for p in seen[0]["problems"]] == [1]  # step 2 is only a consequence
    assert seen[0]["task"] == "demo task"
    assert out["decision"] == "accept"
    assert out["result"].startswith("[Accepted by human reviewer")
    assert "partial result" in out["result"]


def test_human_information_triggers_a_replan(monkeypatch):
    answer_with(monkeypatch, {"action": "provide", "text": "We sell custom cakes in Kochi"})
    out = graph.escalation_node(make_state(PROBLEM))

    assert out["decision"] == "replan"
    assert "We sell custom cakes in Kochi" in out["clarifications"][-1]
    assert out["rounds"] == 1
    assert graph.route_after_escalation({"decision": "replan"}) == "planner"


def test_replan_is_refused_after_the_round_cap(monkeypatch):
    answer_with(monkeypatch, {"action": "provide", "text": "more info"})
    out = graph.escalation_node(make_state(PROBLEM, rounds=graph.MAX_CLARIFY_ROUNDS))

    assert out["decision"] == "abort"
    assert "too many" in out["result"]


def test_abort_stops_the_run(monkeypatch):
    answer_with(monkeypatch, {"action": "abort"})
    out = graph.escalation_node(make_state(PROBLEM))

    assert out["decision"] == "abort"
    assert out["result"] == "Stopped by the human reviewer."
    assert graph.route_after_escalation({"decision": "abort"}) == "remember"


def test_pause_and_resume_through_a_real_checkpointer(caplog, recwarn):
    mini = StateGraph(graph.State)
    mini.add_node("escalation", graph.escalation_node)
    mini.add_edge(START, "escalation")
    mini.add_edge("escalation", END)
    app = mini.compile(checkpointer=graph.make_checkpointer())
    config = {"configurable": {"thread_id": "t1"}}

    out = app.invoke(make_state(PROBLEM), config)
    assert "__interrupt__" in out
    assert out["__interrupt__"][0].value["problems"][0]["step"] == 1

    out = app.invoke(Command(resume={"action": "accept"}), config)
    assert out["decision"] == "accept"
    assert out["result"].startswith("[Accepted by human reviewer")

    # The allow-list must silence the "unregistered type" warning.
    assert "unregistered type" not in caplog.text
    assert not [w for w in recwarn.list if "unregistered type" in str(w.message)]