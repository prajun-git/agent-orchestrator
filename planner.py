import json
from langchain_core.messages import SystemMessage, HumanMessage
from llm import get_llm
from schemas import Plan

SYSTEM_PROMPT = """You are the Supervisor of a multi-agent system.
Break the user's task into a small ordered list of subtasks (2 to 5).
Assign each subtask to exactly one specialist: research, analysis, writing, or code.
Use depends_on to list the ids of subtasks whose output is needed first.

Handling unclear requests:
- If a detail can be reasonably guessed, guess it and record it in assumptions.
- Only use blocking_questions when two different people reading the request would
  do completely different work, so you cannot even tell what is being asked.
- A request like "make my business better" has blocking questions. A request that
  names a clear deliverable and audience usually has none.
Set confidence between 0 and 1 to reflect how likely this plan is to be what the user wants."""

CAP_IF_BLOCKING_QUESTIONS = 0.4
CAP_IF_ASSUMPTIONS = 0.85


def format_memories(memories: list[dict]) -> str:
    """Turn recalled past tasks into a short hints block for the planner prompt."""
    if not memories:
        return ""
    lines = ["Past tasks from this user's history (hints only; the current request may differ, so do not copy blindly):"]
    for m in memories:
        lines.append(f"- Task: {m['task']}")
        lines.append(f"  Outcome: {m['outcome']}")
        if m["plan"]:
            lines.append("  Plan used: " + " -> ".join(m["plan"]))
        for lesson in m["lessons"]:
            lines.append(f"  Lesson: {lesson}")
    return "\n".join(lines)


def _generate_plan(task: str, memories_text: str = "") -> Plan:
    llm = get_llm()
    system = SYSTEM_PROMPT + (f"\n\n{memories_text}" if memories_text else "")
    messages = [SystemMessage(content=system), HumanMessage(content=task)]

    try:
        structured = llm.with_structured_output(Plan, method="function_calling")
        plan = structured.invoke(messages)
        if plan is None:
            raise ValueError("model returned no structured output")
        return plan
    except Exception as e:
        print(f"[structured output failed: {type(e).__name__}] using JSON fallback")
        schema = json.dumps(Plan.model_json_schema())
        messages.append(
            HumanMessage(
                content="Reply with ONLY valid JSON matching this JSON schema, no other text:\n" + schema
            )
        )
        raw = llm.invoke(messages).content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return Plan.model_validate_json(raw)


def create_plan(task: str, memories: list[dict] | None = None) -> Plan:
    plan = _generate_plan(task, format_memories(memories or []))
    # Deterministic guardrail: memory can inform the plan but never bypass this cap.
    if plan.blocking_questions:
        plan.confidence = min(plan.confidence, CAP_IF_BLOCKING_QUESTIONS)
    elif plan.assumptions:
        plan.confidence = min(plan.confidence, CAP_IF_ASSUMPTIONS)
    return plan