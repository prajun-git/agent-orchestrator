from typing import Literal
from pydantic import BaseModel, Field


class Subtask(BaseModel):
    id: int = Field(description="Unique number starting at 1")
    description: str = Field(description="What this subtask must accomplish, in one clear sentence")
    specialist: Literal["research", "analysis", "writing", "code"] = Field(
        description="Which specialist agent handles this subtask"
    )
    depends_on: list[int] = Field(
        default_factory=list,
        description="Ids of subtasks whose output this one needs. Empty if none.",
    )
    expected_output: str = Field(description="What the finished output should look like")
    complexity: Literal["low", "medium", "high"]


class Plan(BaseModel):
    goal: str = Field(description="The user's goal restated in one sentence")
    assumptions: list[str] = Field(
        default_factory=list,
        description="Reasonable assumptions you made in order to proceed. These do NOT block the work.",
    )
    blocking_questions: list[str] = Field(
        default_factory=list,
        description=(
            "Questions that MUST be answered before useful work can start, because without "
            "the answer you would be guessing what the task even is. Empty if sensible "
            "assumptions are enough."
        ),
    )
    subtasks: list[Subtask]
    confidence: float = Field(
        ge=0, le=1,
        description="0 to 1: how confident you are this plan fully solves the task",
    )