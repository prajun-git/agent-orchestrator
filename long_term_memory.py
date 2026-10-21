import json
import time
import uuid

import chromadb


class LongTermMemory:
    """Episodic memory: past tasks, the plan used, and how it turned out. Scoped per user."""

    def __init__(self, embedder, path: str | None = "memory_index", collection: str = "task_memory"):
        self.embedder = embedder
        if path is None:  # in-memory, used by tests and demos
            self.client = chromadb.EphemeralClient()
            collection = f"{collection}_{uuid.uuid4().hex[:8]}"
        else:
            self.client = chromadb.PersistentClient(path=path)
        self.col = self.client.get_or_create_collection(collection)

    def remember(self, user_id: str, task: str, plan_outline: list[str], outcome: str, lessons: list[str]) -> str:
        memory_id = uuid.uuid4().hex
        self.col.add(
            ids=[memory_id],
            embeddings=[self.embedder.embed_query(task)],
            documents=[task],
            metadatas=[{  # Chroma metadata must be simple values, so lists are stored as JSON text
                "user_id": user_id,
                "outcome": outcome,
                "plan": json.dumps(plan_outline),
                "lessons": json.dumps(lessons),
                "created_at": time.time(),
                "access_count": 0,
            }],
        )
        return memory_id

    def recall(self, user_id: str, task: str, k: int = 3, max_distance: float = 1.0) -> list[dict]:
        """Past tasks of this user that are similar to `task`, closest first."""
        matching = len(self.col.get(where={"user_id": user_id}, include=[])["ids"])
        if matching == 0:
            return []

        res = self.col.query(
            query_embeddings=[self.embedder.embed_query(task)],
            n_results=min(k, matching),
            where={"user_id": user_id},
        )

        found = []
        for i, memory_id in enumerate(res["ids"][0]):
            distance = res["distances"][0][i]
            if distance > max_distance:
                continue
            meta = res["metadatas"][0][i]
            found.append({
                "id": memory_id,
                "task": res["documents"][0][i],
                "distance": round(distance, 3),
                "outcome": meta["outcome"],
                "plan": json.loads(meta["plan"]),
                "lessons": json.loads(meta["lessons"]),
            })
            # Memories that keep being useful gain importance (used for scoring and expiry later).
            self.col.update(ids=[memory_id], metadatas=[{**meta, "access_count": meta["access_count"] + 1}])
        return found

    def forget(self, memory_id: str) -> None:
        self.col.delete(ids=[memory_id])

    def delete_user(self, user_id: str) -> int:
        """Delete everything stored for one user (a data-deletion request). Returns how many."""
        ids = self.col.get(where={"user_id": user_id}, include=[])["ids"]
        if ids:
            self.col.delete(ids=ids)
        return len(ids)


def summarize_run(plan, results: dict) -> tuple[list[str], str, list[str]]:
    """Turn a finished run into (plan outline, outcome, lessons) for storage.
    The lessons are crude, rule-generated notes for now; an LLM can write better ones later."""
    outline = [f"{s.specialist}: {s.description}" for s in plan.subtasks]
    statuses = [results[s.id]["status"] for s in plan.subtasks if s.id in results]

    if statuses and all(x == "done" for x in statuses):
        outcome = "success"
    elif any(x == "done" for x in statuses):
        outcome = "partial"
    else:
        outcome = "failed"

    lessons = [
        f"step {s.id} ({s.specialist}) ended as {results[s.id]['status']}"
        for s in plan.subtasks
        if s.id in results and results[s.id]["status"] != "done"
    ]
    return outline, outcome, lessons