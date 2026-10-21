from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from llm import get_llm
from registry import ToolRegistry

MAX_STEPS = 5

COMMON_RULES = (
    "\n\nRules for every specialist: your final reply IS the deliverable that the next step "
    "will receive. Write the complete result in your reply text. Never describe what you did "
    "or what a tool call does; produce the content itself. Use only the task and the context "
    "provided. If information you need is missing, say so instead of inventing it."
)

SPECIALIST_PROMPTS = {
    "research": (
        "You are the Research specialist in a multi-agent system. "
        "Use the search_knowledge tool to look up facts in the internal knowledge base. "
        "Never paste the task text into the search. Write each query as a full, self-contained "
        "question that names the topic and the organization, for example "
        "'What is the delivery policy of Acme Bakery?'. Bare keywords and instructions "
        "like 'Research X' search badly. "
        "If a search finds nothing relevant, rephrase and search once more before concluding "
        "that the knowledge base has no information; only say so after two failed searches. "
        "Base your answer on what the tool returns and mention the source file names. "
        "State only facts that appear in the tool output. If you add anything from your own "
        "general knowledge, label it explicitly as 'general knowledge, not from the knowledge base'."
    ),
    "analysis": (
        "You are the Analysis specialist in a multi-agent system. "
        "Reason carefully about the context from earlier steps. Use the calculator tool for "
        "any arithmetic and search_knowledge for facts that are not in the context. "
        "Clearly separate facts from your interpretation of them."
    ),
    "writing": (
        "You are the Writing specialist in a multi-agent system. "
        "Write clear, well-organized text for the audience named in the task, using only the "
        "facts in the context from earlier steps. Do not invent policies, numbers or details. "
        "Put each fact only under the heading it belongs to; leave out facts that do not "
        "belong to the topic. You can read files but you cannot write or save files. "
        "Reply with the finished text itself, as plain prose or bullet points, never as JSON."
    ),
    "code": (
        "You are the Code specialist in a multi-agent system. "
        "Write the requested code in your reply. You currently have no way to run code, so "
        "state clearly that your code has not been executed or tested."
    ),
}


def run_specialist(
    specialist: str, subtask: str, registry: ToolRegistry, context: str = "", feedback: str = ""
) -> dict:
    # Only the tools this specialist is allowed to use are shown to the model.
    specs = registry.tools_for(specialist)
    lc_tools = [
        StructuredTool.from_function(func=s.func, name=s.name, description=s.description)
        for s in specs
    ]
    llm = get_llm().bind_tools(lc_tools) if lc_tools else get_llm()

    role = SPECIALIST_PROMPTS.get(
        specialist, f"You are the {specialist} specialist in a multi-agent system."
    )
    user = subtask
    if context:
        user += f"\n\nContext from earlier steps:\n{context}"
    if feedback:
        user += (
            "\n\nYour previous attempt was rejected by a reviewer for these reasons:\n"
            f"{feedback}\nFix every problem in your new answer."
        )
    messages = [SystemMessage(content=role + COMMON_RULES), HumanMessage(content=user)]

    evidence: list[str] = []  # successful tool outputs, used by the reviewer's groundedness check

    for step in range(1, MAX_STEPS + 1):
        ai = llm.invoke(messages)
        messages.append(ai)

        if not ai.tool_calls:  # the model is done
            return {"success": True, "output": ai.content, "steps": step, "evidence": "\n".join(evidence)}

        for call in ai.tool_calls:
            # Every tool call still goes through the registry gate.
            result = registry.call(call["name"], specialist, **call["args"])
            if result.success:
                evidence.append(str(result.output))
            content = str(result.output) if result.success else f"ERROR: {result.error}"
            messages.append(ToolMessage(content=content, tool_call_id=call["id"]))

    return {
        "success": False,
        "output": "Stopped: too many steps without a final answer.",
        "steps": MAX_STEPS,
        "evidence": "\n".join(evidence),
    }