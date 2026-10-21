import json
from typing import Any


class WorkingMemory:
    """Scratchpad for ONE task, shared by all agents. Everything stored must be JSON-serializable,
    so a Redis-backed version with the same methods can replace this class later."""

    def __init__(self, task_id: str = "adhoc") -> None:
        self.task_id = task_id
        self._data: dict[str, Any] = {}
        self._errors: list[dict] = []

    def set(self, key: str, value: Any) -> None:
        json.dumps(value)  # raises TypeError for anything Redis could not store
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def record_step(self, step_id: int, status: str, output: str) -> None:
        self.set(f"step:{step_id}", {"status": status, "output": output})

    def step_output(self, step_id: int) -> str:
        return self.get(f"step:{step_id}", {}).get("output", "")

    def log_error(self, step_id: int, message: str) -> None:
        self._errors.append({"step": step_id, "message": message})

    @property
    def errors(self) -> list[dict]:
        return list(self._errors)

    def snapshot(self) -> dict:
        """A copy of everything, used later to feed long-term memory."""
        return json.loads(json.dumps({"task_id": self.task_id, "data": self._data, "errors": self._errors}))

    def clear(self) -> None:
        self._data.clear()
        self._errors.clear()