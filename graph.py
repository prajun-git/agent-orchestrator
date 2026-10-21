import os
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from executor import execute_plan
from long_term_memory import LongTermMemory, summarize_run
from planner import create_plan
from rag import get_embeddings
from schemas import Plan
from tools_builtin import build_registry
from working_memory import WorkingMemory

MAX_CLARIFY_ROUNDS = 2
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

    get_memory().remember(state["user_id"], state["task"], outline, outcome, lessons)
    print(f"[memory] saved this run as '{outcome}'")
    return {}


graph = StateGraph(State)
graph.add_node("recall", recall_node)
graph.add_node("planner", planner_node)
graph.add_node("clarify", clarify_node)
graph.add_node("executor", executor_node)
graph.add_node("give_up", give_up_node)
graph.add_node("remember", remember_node)

graph.add_edge(START, "recall")
graph.add_edge("recall", "planner")
graph.add_conditional_edges("planner", route_after_plan, ["clarify", "executor", "give_up"])
graph.add_edge("clarify", "planner")
graph.add_edge("executor", "remember")
graph.add_edge("give_up", "remember")
graph.add_edge("remember", END)

app = graph.compile()

if __name__ == "__main__":
    task = input("Task: ")
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
        }
    )
    print("\n=== RESULT ===")
    print(out["result"])
    print("\n=== TOOL CALLS ===")
    for e in out["tool_log"]:
        print(f"  {e['specialist']:9} {e['tool']:17} success={e['success']}  {e['latency_ms']} ms")
        print(f"      inputs: {e['inputs']}")
        print(f"      output: {e['output'][:120]!r}")