from pathlib import Path

docs = {
    "vector_databases.md": """# Vector databases
A vector database stores embeddings, which are lists of numbers that represent the meaning of text, images or other data.
Instead of matching exact keywords, it finds items whose vectors are closest to the query vector.
Most use approximate nearest neighbor search, such as HNSW indexes, to trade a little accuracy for much faster lookups.
Common choices include Chroma, Qdrant, Pinecone, Weaviate and pgvector, a PostgreSQL extension.
""",
    "microservices.md": """# Microservices
Microservices split an application into small services that are deployed independently.
Benefits: teams can ship independently, services scale separately, and one failure is easier to isolate.
Costs: network calls replace function calls, debugging spans many services, and data consistency becomes harder.
Small teams often do better starting with a well-structured monolith.
""",
    "acme_bakery_policy.md": """# Acme Bakery internal policy
Custom cake orders must be placed at least 72 hours in advance.
Refunds are available within 7 days for unopened packaged goods, but never for custom cakes.
Delivery is free for orders above 1500 rupees within 10 km of the shop.
The shop closes at 8 pm on weekdays and 10 pm on Saturdays. Sunday is closed.
""",
}

folder = Path("knowledge")
folder.mkdir(exist_ok=True)
for name, text in docs.items():
    (folder / name).write_text(text, encoding="utf-8")
print(f"Created {len(docs)} documents in {folder}/")