# Vector databases
A vector database stores embeddings, which are lists of numbers that represent the meaning of text, images or other data.
Instead of matching exact keywords, it finds items whose vectors are closest to the query vector.
Most use approximate nearest neighbor search, such as HNSW indexes, to trade a little accuracy for much faster lookups.
Common choices include Chroma, Qdrant, Pinecone, Weaviate and pgvector, a PostgreSQL extension.
