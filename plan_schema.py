from typing import Literal
from pydantic import BaseModel, Field, ValidationError


class Subtask(BaseModel):
    id: int
    description: str
    specialist: Literal["research", "analysis", "writing", "code"]
    depends_on: list[int] = Field(default_factory=list)
    expected_output: str
    complexity: Literal["low", "medium", "high"]


class Plan(BaseModel):
    goal: str
    subtasks: list[Subtask]
    confidence: float = Field(ge=0, le=1)


good = {
    "goal": "Compare vector databases",
    "confidence": 0.85,
    "subtasks": [
        {"id": 1, "description": "Search for top vector databases", "specialist": "research",
         "expected_output": "list of databases with features", "complexity": "low"},
        {"id": 2, "description": "Write a comparison summary", "specialist": "writing",
         "depends_on": [1], "expected_output": "300-word summary", "complexity": "medium"},
    ],
}

bad = {
    "goal": "Compare vector databases",
    "confidence": 1.7,
    "subtasks": [
        {"id": 1, "description": "Cook dinner", "specialist": "chef",
         "expected_output": "food", "complexity": "low"},
    ],
}

if __name__ == "__main__":
    plan = Plan(**good)
    print("GOOD PLAN OK:", plan.subtasks[1].depends_on)

    try:
        Plan(**bad)
    except ValidationError as e:
        print("\nBAD PLAN REJECTED:")
        print(e)
