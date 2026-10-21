from rag import build_index, search_knowledge

print("Indexed chunks:", build_index())

questions = [
    "How far in advance do I need to order a custom cake?",
    "What are the downsides of splitting an app into many services?",
    "What is the capital of France?",  # not in our documents
]

for q in questions:
    print("\n" + "=" * 60)
    print("QUESTION:", q)
    print(search_knowledge(q, k=2))