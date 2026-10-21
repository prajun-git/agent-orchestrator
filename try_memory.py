from long_term_memory import LongTermMemory
from rag import get_embeddings

mem = LongTermMemory(get_embeddings(), path=None)  # in-memory demo, nothing saved to disk

past = [
    ("Summarize the refund and delivery policies of Acme Bakery for a new employee",
     ["research: refund policy", "research: delivery policy", "writing: summary"], "success", []),
    ("Compare the top 3 vector databases and write a summary for a CTO",
     ["research: databases", "analysis: compare", "writing: summary"], "success", []),
    ("Make my business better",
     [], "failed", ["planner had blocking questions"]),
]
for task, plan, outcome, lessons in past:
    mem.remember("me", task, plan, outcome, lessons)

queries = [
    "What are the refund and delivery rules at Acme Bakery? Write it up for a new hire.",  # similar
    "Explain the opening hours of Acme Bakery",                                           # same topic area
    "Compare pgvector and Qdrant for a startup",                                           # similar to task 2
    "Help me improve my business",                                                         # similar to task 3
    "What is the capital of France?",                                                      # unrelated
    "Write a python function that sorts a list",                                           # unrelated
]

for q in queries:
    hits = mem.recall("me", q, k=1, max_distance=99)  # max_distance=99 -> print the distance of the best match
    best = hits[0]
    print(f"{best['distance']:.2f}  {q}\n        -> {best['task'][:60]}  [{best['outcome']}]")