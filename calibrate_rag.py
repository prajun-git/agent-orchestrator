from rag import build_index, _get_store

build_index()
store = _get_store()

relevant = [
    "refund policy",
    "refunds",
    "How far in advance do I need to order a custom cake?",
    "is delivery free",
    "when does the shop close on Saturday",
    "what is a vector database",
    "HNSW",
    "pgvector",
    "downsides of microservices",
    "should a small team use a monolith",
]

irrelevant = [
    "What is the capital of France?",
    "who won the football world cup",
    "python list comprehension",
    "weather in Kochi tomorrow",
    "how to bake sourdough bread",
    "best pizza recipe",
]


def best_distance(query: str) -> float:
    doc, score = store.similarity_search_with_score(query, k=1)[0]
    return score


print("RELEVANT")
for q in relevant:
    print(f"  {best_distance(q):.2f}  {q}")

print("\nIRRELEVANT")
for q in irrelevant:
    print(f"  {best_distance(q):.2f}  {q}")