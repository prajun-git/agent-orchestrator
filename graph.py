import os
import uuid
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from executor import execute_plan
from long_term_memory import LongTermMemory, summarize_run
from planner import create_plan
from rag import get_embeddings
from schemas import Plan
from tools_builtin import build_registry
from working_memory import WorkingMemory

MAX_CLARIFY_ROUNDS = 2
ROOT_PROBLEMS = {"no_data", "rejected", "failed"}  # "skipped" is only a consequence of these
_memory = None


def get_memory() -> LongTermMemory:
    """Create the long-term memory once (it needs the embedding model, which loads slowly)."""
    global _memory
    if _memory is None:
        _memory = LongTermMemory(get_embeddings(), path="memory_index")
    return _memory


class State(TypedDict):
    user_id: str
    task: str
    memories: list
    clarifications: list[str]
    plan: Plan | None
    rounds: int
    step_results: dict
    tool_log: list
    result: str
    decision: str


def recall_node(state: State) -> dict:
    memories = get_memory().recall(state["user_id"], state["task"])
    print(f"[memory] recalled {len(memories)} similar past task(s)")
    for m in memories:
        print(f"  - ({m['outcome']}, distance {m['distance']}) {m['task'][:70]}")
    return {"memories": memories}


def planner_node(state: State) -> dict:
    task = state["task"]
    if state["clarifications"]:
        task += "\n\nUser clarifications:\n" + "\n".join(f"- {c}" for c in state["clarifications"])
    plan = create_plan(task, state["memories"])
    print(f"[planner] confidence={plan.confidence}, subtasks={len(plan.subtasks)}")
    return {"plan": plan}


def route_after_plan(state: State) -> str:
    plan = state["plan"]
    if plan.blocking_questions:
        return "clarify" if state["rounds"] < MAX_CLARIFY_ROUNDS else "give_up"
    if not plan.subtasks:
        return "give_up"
    return "executor"


def clarify_node(state: State) -> dict:
    print("\nThe supervisor needs more information:")
    answers = []
    for q in state["plan"].blocking_questions:
        a = input(f"  {q}\n  > ")
        answers.append(f"{q} -> {a}")
    return {
        "clarifications": state["clarifications"] + answers,
        "rounds": state["rounds"] + 1,
    }


def executor_node(state: State) -> dict:
    registry = build_registry()  # one registry per task, so the log covers the whole run
    working = WorkingMemory()
    results = execute_plan(state["plan"], registry, working)

    sections = []
    for s in state["plan"].subtasks:
        r = results[s.id]
        sections.append(f"### Step {s.id} [{s.specialist}] {s.description}  ({r['status']})\n{r['output']}")
    return {"step_results": results, "tool_log": registry.log, "result": "\n\n".join(sections)}


def escalation_node(state: State) -> dict:
    """Pause for a human when a step ended without usable output.
    NOTE: on resume this node runs again from its first line, so everything before
    interrupt() must be side-effect free."""
    plan, results = state["plan"], state["step_results"]
    problems = [
        {
            "step": s.id,
            "description": s.description,
            "status": results[s.id]["status"],
            "detail": str(results[s.id]["output"])[:200],
        }
        for s in plan.subtasks
        if s.id in results and results[s.id]["status"] in ROOT_PROBLEMS
    ]
    if not problems:
        return {"decision": "continue"}

    completed = [
        {"step": s.id, "description": s.description}
        for s in plan.subtasks
        if s.id in results and results[s.id]["status"] == "done"
    ]
    answer = interrupt({"task": state["task"], "problems": problems, "completed": completed})

    action = answer.get("action")
    if action == "provide" and state["rounds"] < MAX_CLARIFY_ROUNDS:
        return {
            "decision": "replan",
            "clarifications": state["clarifications"]
            + [f"Information provided by the human reviewer: {answer.get('text', '')}"],
            "rounds": state["rounds"] + 1,
        }
    if action == "accept":
        flagged = ", ".join(f"step {p['step']} ({p['status']})" for p in problems)
        return {
            "decision": "accept",
            "result": f"[Accepted by human reviewer with unresolved problems: {flagged}]\n\n{state['result']}",
        }
    reason = "Stopped by the human reviewer." if action == "abort" else "Stopped: too many review rounds."
    return {"decision": "abort", "result": reason}


def route_after_escalation(state: State) -> str:
    return "planner" if state["decision"] == "replan" else "remember"


def give_up_node(state: State) -> dict:
    return {"result": "Stopped: the task is still too unclear after asking for clarification."}


def remember_node(state: State) -> dict:
    """Store the finished (or abandoned) run so future plans can learn from it."""
    plan = state["plan"]
    if state["step_results"]:
        outline, outcome, lessons = summarize_run(plan, state["step_results"])
    else:  # gave up before executing anything
        outline, outcome = [], "failed"
        questions = "; ".join(plan.blocking_questions) if plan else "unknown"
        lessons = [f"request stayed too vague after clarification; unanswered questions: {questions}"]

    if state.get("decision") in {"accept", "abort"}:
        lessons = lessons + [f"a human reviewer stepped in and chose to {state['decision']}"]

    get_memory().remember(state["user_id"], state["task"], outline, outcome, lessons)
    print(f"[memory] saved this run as '{outcome}'")
    return {}


graph = StateGraph(State)
graph.add_node("recall", recall_node)
graph.add_node("planner", planner_node)
graph.add_node("clarify", clarify_node)
graph.add_node("executor", executor_node)
graph.add_node("escalation", escalation_node)
graph.add_node("give_up", give_up_node)
graph.add_node("remember", remember_node)

graph.add_edge(START, "recall")
graph.add_edge("recall", "planner")
graph.add_conditional_edges("planner", route_after_plan, ["clarify", "executor", "give_up"])
graph.add_edge("clarify", "planner")
graph.add_edge("executor", "escalation")
graph.add_conditional_edges("escalation", route_after_escalation, ["planner", "remember"])
graph.add_edge("give_up", "remember")
graph.add_edge("remember", END)

def make_checkpointer() -> MemorySaver:
    """In-memory checkpointer that is explicitly allowed to restore our own classes."""
    serde = JsonPlusSerializer(allowed_msgpack_modules=[("schemas", "Plan"), ("schemas", "Subtask")])
    return MemorySaver(serde=serde)


app = graph.compile(checkpointer=make_checkpointer())

def ask_human(payload: dict) -> dict:
    """Terminal version of the review screen. A web UI would show the same payload."""
    print("\n=== HUMAN REVIEW NEEDED ===")
    print(f"Task: {payload['task']}")
    print("Problems:")
    for p in payload["problems"]:
        print(f"  step {p['step']} [{p['status']}]: {p['description']}")
        print(f"      {p['detail']}")
    if payload["completed"]:
        print("Completed steps: " + ", ".join(str(c["step"]) for c in payload["completed"]))
    print("\nChoose: 'accept' (deliver as is), 'abort', or type the missing information to retry")
    while True:
        choice = input("> ").strip()
        if choice.lower() in {"accept", "a"}:
            return {"action": "accept"}
        if choice.lower() in {"abort", "x"}:
            return {"action": "abort"}
        if choice:
            return {"action": "provide", "text": choice}


if __name__ == "__main__":
    task = input("Task: ")
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}  # identifies this run for pause/resume
    out = app.invoke(
        {
            "user_id": os.environ.get("AGENT_USER_ID", "me"),
            "task": task,
            "memories": [],
            "clarifications": [],
            "plan": None,
            "rounds": 0,
            "step_results": {},
            "tool_log": [],
            "result": "",
            "decision": "",
        },
        config,
    )
    while "__interrupt__" in out:  # the graph is paused and waiting for a person
        answer = ask_human(out["__interrupt__"][0].value)
        out = app.invoke(Command(resume=answer), config)

    print("\n=== RESULT ===")
    print(out["result"])
    print("\n=== TOOL CALLS ===")
    for e in out["tool_log"]:
        print(f"  {e['specialist']:9} {e['tool']:17} success={e['success']}  {e['latency_ms']} ms")
        print(f"      inputs: {e['inputs']}")
        print(f"      output: {e['output'][:120]!r}")