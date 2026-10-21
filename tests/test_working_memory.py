import pytest
from working_memory import WorkingMemory


def test_set_and_get():
    mem = WorkingMemory("t1")
    mem.set("plan", {"goal": "demo"})
    assert mem.get("plan") == {"goal": "demo"}
    assert mem.get("missing", "fallback") == "fallback"


def test_non_serializable_values_are_rejected():
    with pytest.raises(TypeError):
        WorkingMemory().set("bad", {1, 2, 3})  # a set is not JSON-serializable


def test_step_records_and_errors():
    mem = WorkingMemory()
    mem.record_step(1, "done", "refunds within 7 days")
    mem.log_error(2, "step ended as rejected")
    assert mem.step_output(1) == "refunds within 7 days"
    assert mem.step_output(99) == ""
    assert mem.errors == [{"step": 2, "message": "step ended as rejected"}]


def test_tasks_do_not_share_memory():
    a, b = WorkingMemory("a"), WorkingMemory("b")
    a.set("k", 1)
    assert b.get("k") is None


def test_snapshot_is_a_copy_and_clear_empties_everything():
    mem = WorkingMemory("t1")
    mem.set("k", [1, 2])
    snap = mem.snapshot()
    mem.clear()
    assert snap["data"]["k"] == [1, 2]
    assert mem.get("k") is None and mem.errors == []