import hashlib
import math

import pytest
from long_term_memory import LongTermMemory, summarize_run
from schemas import Plan, Subtask


class FakeEmbedder:
    """Bag-of-words hashing: tasks that share words get nearby vectors. No model download needed."""

    def embed_query(self, text: str) -> list[float]:
        vec = [0.0] * 64
        for word in text.lower().split():
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


STORED_TASK = "summarize refund policy of acme bakery"
SIMILAR_TASK = "summarize the refund policy of acme bakery shop"
UNRELATED_TASK = "weather in kochi tomorrow"


@pytest.fixture
def mem() -> LongTermMemory:
    m = LongTermMemory(FakeEmbedder(), path=None)
    m.remember("u1", STORED_TASK, ["research: refunds", "writing: summary"], "success", [])
    m.remember("u1", "compare vector databases for a cto", ["research: dbs"], "partial",
               ["step 2 (writing) ended as skipped"])
    return m


def test_recall_finds_the_similar_task(mem):
    found = mem.recall("u1", SIMILAR_TASK)
    assert found and found[0]["task"] == STORED_TASK
    assert found[0]["outcome"] == "success"
    assert found[0]["plan"] == ["research: refunds", "writing: summary"]


def test_unrelated_task_is_not_recalled(mem):
    assert mem.recall("u1", UNRELATED_TASK) == []


def test_memory_is_scoped_per_user(mem):
    assert mem.recall("someone_else", SIMILAR_TASK) == []


def test_recall_counts_access(mem):
    memory_id = mem.recall("u1", SIMILAR_TASK)[0]["id"]
    mem.recall("u1", SIMILAR_TASK)
    assert mem.col.get(ids=[memory_id])["metadatas"][0]["access_count"] == 2


def test_delete_user_removes_only_that_users_memories(mem):
    mem.remember("u2", STORED_TASK, ["x"], "success", [])
    assert mem.delete_user("u1") == 2
    assert mem.recall("u1", SIMILAR_TASK) == []
    assert mem.recall("u2", SIMILAR_TASK) != []


def test_forget_single_memory(mem):
    memory_id = mem.recall("u1", SIMILAR_TASK)[0]["id"]
    mem.forget(memory_id)
    assert mem.recall("u1", SIMILAR_TASK) == []


def _plan() -> Plan:
    return Plan(
        goal="demo",
        subtasks=[
            Subtask(id=1, description="Research", specialist="research", expected_output="x", complexity="low"),
            Subtask(id=2, description="Write", specialist="writing", depends_on=[1],
                    expected_output="x", complexity="low"),
        ],
        confidence=0.9,
    )


@pytest.mark.parametrize(
    "statuses,outcome,n_lessons",
    [
        (("done", "done"), "success", 0),
        (("done", "skipped"), "partial", 1),
        (("failed", "skipped"), "failed", 2),
    ],
)
def test_summarize_run_outcomes(statuses, outcome, n_lessons):
    results = {1: {"status": statuses[0]}, 2: {"status": statuses[1]}}
    outline, result_outcome, lessons = summarize_run(_plan(), results)
    assert outline == ["research: Research", "writing: Write"]
    assert result_outcome == outcome
    assert len(lessons) == n_lessons