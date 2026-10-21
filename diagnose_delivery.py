from rag import _get_store, MAX_DISTANCE

queries = [
    "delivery",
    "delivery policy",
    "Research delivery policy",
    "Acme Bakery delivery policy",
    "When is delivery free?",
    "What is the delivery policy of Acme Bakery?",
]

store = _get_store()
for q in queries:
    doc, score = store.similarity_search_with_score(q, k=1)[0]
    verdict = "kept" if score <= MAX_DISTANCE else "REJECTED"
    print(f"{score:.2f}  {verdict:8}  {q}")