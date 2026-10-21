import pytest
from registry import ToolRegistry, ToolSpec
from tools_builtin import calculator, read_file


def make_registry(limit: int = 30) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolSpec(name="echo", description="echo text", func=lambda text: text,
                          allowed_specialists={"writing"}, max_calls_per_minute=limit))
    reg.register(ToolSpec(name="boom", description="always fails", func=lambda: 1 / 0,
                          allowed_specialists={"writing"}))
    return reg


def test_allowed_call_succeeds_and_is_logged():
    reg = make_registry()
    result = reg.call("echo", "writing", text="hi")
    assert result.success and result.output == "hi"
    assert reg.log[-1]["tool"] == "echo" and reg.log[-1]["success"] is True


def test_wrong_specialist_is_denied():
    result = make_registry().call("echo", "research", text="hi")
    assert not result.success and "not allowed" in result.error


def test_unknown_tool_is_reported():
    result = make_registry().call("hack_the_planet", "writing")
    assert not result.success and "unknown tool" in result.error


def test_rate_limit_blocks_extra_calls():
    reg = make_registry(limit=3)
    outcomes = [reg.call("echo", "writing", text="x").success for _ in range(5)]
    assert outcomes == [True, True, True, False, False]


def test_tool_exception_is_returned_as_data():
    result = make_registry().call("boom", "writing")
    assert not result.success and "ZeroDivisionError" in result.error


def test_calculator_computes():
    assert calculator("(12 + 8) * 3 / 4") == 15


@pytest.mark.parametrize("expression", ["__import__('os').listdir('.')", "2**1000", "open('x')"])
def test_calculator_rejects_unsafe_input(expression):
    with pytest.raises(ValueError):
        calculator(expression)


def test_file_sandbox_blocks_path_escape():
    with pytest.raises(PermissionError):
        read_file("../secret.txt")