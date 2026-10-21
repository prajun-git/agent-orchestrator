from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    task: str
    plan: list[str]
    result: str


def planner(state: State) -> dict:
    # Later this will be an LLM call. For now it is a fake plan.
    return {"plan": [f"research: {state['task']}", f"write summary: {state['task']}"]}


def executor(state: State) -> dict:
    return {"result": " | ".join(f"done -> {step}" for step in state["plan"])}


def route_after_plan(state: State) -> str:
    # Conditional edge: only run the executor if a plan exists
    return "executor" if state["plan"] else END


graph = StateGraph(State)
graph.add_node("planner", planner)
graph.add_node("executor", executor)

graph.add_edge(START, "planner")
graph.add_conditional_edges("planner", route_after_plan, ["executor", END])
graph.add_edge("executor", END)

app = graph.compile()

if __name__ == "__main__":
    out = app.invoke({"task": "compare vector databases", "plan": [], "result": ""})
    print(out)
