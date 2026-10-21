import sys
from long_term_memory import LongTermMemory

user = sys.argv[1] if len(sys.argv) > 1 else "me"
count = LongTermMemory(embedder=None, path="memory_index").delete_user(user)
print(f"Deleted {count} memories for user '{user}'")