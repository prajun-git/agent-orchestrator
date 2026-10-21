import re
from dataclasses import dataclass, field


@dataclass
class Review:
    approved: bool
    issues: list[str] = field(default_factory=list)


def _numbers(text: str) -> set[str]:
    text = re.sub(r"(?m)^\s*\d+[.)]\s", "", text)  # ignore list numbering like "1. "
    text = text.replace(",", "")  # treat 1,500 and 1500 as the same number
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def review_output(specialist: str, output: str, evidence: str, task: str, context: str) -> Review:
    """Rule-based review. `evidence` is the text returned by this step's successful tool calls."""
    issues: list[str] = []
    text = output.strip()

    if len(text) < 20:
        issues.append("The output is empty or too short to be a real deliverable.")

    if specialist != "code" and text.startswith(("{", "[")):
        issues.append("The output is raw JSON or tool-call text. Write the finished deliverable as plain text.")

    if "ERROR" in text or text.startswith("Stopped:"):
        issues.append("The output contains an error message instead of a result.")

    if specialist == "research" and not evidence.strip():
        issues.append("The research step did not retrieve anything from the knowledge base. Search before answering.")

    if specialist in {"research", "analysis", "writing"}:
        sources = f"{evidence}\n{task}\n{context}"
        ungrounded = _numbers(output) - _numbers(sources)
        if ungrounded:
            issues.append(
                f"These numbers do not appear in the task, context or tool output: {sorted(ungrounded)}. "
                "Use only numbers from the provided sources."
            )

    return Review(approved=not issues, issues=issues)