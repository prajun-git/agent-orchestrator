import ast
import operator
from pathlib import Path

from registry import ToolRegistry, ToolSpec
from rag import search_knowledge

# ---------- calculator (no eval!) ----------
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def calculator(expression: str) -> float:
    """Evaluate a basic arithmetic expression like '(12 + 8) * 3 / 4'."""

    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("only basic arithmetic is allowed")

    return ev(ast.parse(expression, mode="eval").body)


# ---------- sandboxed file tools ----------
WORKSPACE = Path("workspace").resolve()
WORKSPACE.mkdir(exist_ok=True)


def _safe_path(filename: str) -> Path:
    path = (WORKSPACE / filename).resolve()
    if not path.is_relative_to(WORKSPACE):
        raise PermissionError("path is outside the workspace")
    return path


def read_file(filename: str) -> str:
    """Read a text file from the workspace folder."""
    return _safe_path(filename).read_text(encoding="utf-8")


def write_file(filename: str, content: str) -> str:
    """Write text to a file in the workspace folder."""
    if " " in filename or "." not in Path(filename).name:
        raise ValueError("filename must look like 'notes.txt' (no spaces, with an extension)")
    path = _safe_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} characters to {filename}"


# ---------- build the registry ----------
def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="calculator",
            description="Evaluate a basic arithmetic expression.",
            func=calculator,
            allowed_specialists={"analysis", "code"},
            max_calls_per_minute=30,
        )
    )
    registry.register(
        ToolSpec(
            name="read_file",
            description="Read a text file from the workspace folder.",
            func=read_file,
            allowed_specialists={"research", "analysis", "writing", "code"},
        )
    )
    registry.register(
        ToolSpec(
            name="write_file",
            description="Write text to a file in the workspace folder.",
            func=write_file,
            allowed_specialists={"code"},
            max_calls_per_minute=10,
        )
    )
    registry.register(
        ToolSpec(
            name="search_knowledge",
            description="Search the internal knowledge base and return the most relevant passages.",
            func=search_knowledge,
            allowed_specialists={"research", "analysis"},
            max_calls_per_minute=20,
        )
    )
    return registry