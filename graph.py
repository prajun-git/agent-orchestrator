from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from executor import execute_plan
from planner import create_plan
from schemas import Plan
from tools_builtin import build_registry

MAX_CLARIFY_ROUNDS = 2


class State(TypedDict):
    task: str
    clarifications: list[str]
    plan: Plan | None
    rounds: int
    step_results: dict
    tool_log: list
    result: str


def planner_node(state: State) -> dict:
    task = state["task"]
    if state["clarifications"]:
        task += "\n\nUser clarifications:\n" + "\n".join(f"- {c}" for c in state["clarifications"])
    plan = create_plan(task)
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
    results = execute_plan(state["plan"], registry)

    sections = []
    for s in state["plan"].subtasks:
        r = results[s.id]
        sections.append(f"### Step {s.id} [{s.specialist}] {s.description}  ({r['status']})\n{r['output']}")
    return {"step_results": results, "tool_log": registry.log, "result": "\n\n".join(sections)}


def give_up_node(state: State) -> dict:
    return {"result": "Stopped: the task is still too unclear after asking for clarification."}


graph = StateGraph(State)
graph.add_node("planner", planner_node)
graph.add_node("clarify", clarify_node)
graph.add_node("executor", executor_node)
graph.add_node("give_up", give_up_node)

graph.add_edge(START, "planner")
graph.add_conditional_edges("planner", route_after_plan, ["clarify", "executor", "give_up"])
graph.add_edge("clarify", "planner")
graph.add_edge("executor", END)
graph.add_edge("give_up", END)

app = graph.compile()

if __name__ == "__main__":
    task = input("Task: ")
    out = app.invoke(
        {
            "task": task,
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