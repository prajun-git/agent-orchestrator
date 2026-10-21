import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolSpec:
    name: str
    description: str
    func: Callable[..., Any]
    allowed_specialists: set[str]
    max_calls_per_minute: int = 30


@dataclass
class ToolResult:
    success: bool
    output: Any = None
    error: str | None = None
    latency_ms: float = 0.0


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._recent_calls: dict[str, deque] = {}
        self.log: list[dict] = []

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec
        self._recent_calls[spec.name] = deque()

    def tools_for(self, specialist: str) -> list[ToolSpec]:
        return [t for t in self._tools.values() if specialist in t.allowed_specialists]

    def call(self, tool_name: str, specialist: str, **kwargs: Any) -> ToolResult:
        start = time.perf_counter()
        result = self._run(tool_name, specialist, kwargs)
        result.latency_ms = round((time.perf_counter() - start) * 1000, 2)

        self.log.append(
            {
                "time": time.strftime("%H:%M:%S"),
                "tool": tool_name,
                "specialist": specialist,
                "inputs": kwargs,
                "output": str(result.output)[:200],
                "success": result.success,
                "error": result.error,
                "latency_ms": result.latency_ms,
            }
        )
        return result

    def _run(self, tool_name: str, specialist: str, kwargs: dict) -> ToolResult:
        spec = self._tools.get(tool_name)
        if spec is None:
            return ToolResult(False, error=f"unknown tool '{tool_name}'")

        if specialist not in spec.allowed_specialists:
            return ToolResult(False, error=f"'{specialist}' is not allowed to use '{tool_name}'")

        # Sliding-window rate limit: keep only calls from the last 60 seconds
        now = time.monotonic()
        window = self._recent_calls[tool_name]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= spec.max_calls_per_minute:
            return ToolResult(False, error=f"rate limit hit for '{tool_name}'")
        window.append(now)

        try:
            return ToolResult(True, output=spec.func(**kwargs))
        except Exception as e:
            return ToolResult(False, error=f"{type(e).__name__}: {e}")