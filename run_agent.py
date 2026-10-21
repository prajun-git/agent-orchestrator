from agent import run_specialist
from tools_builtin import build_registry

questions = [
    "What is the refund policy for custom cakes?",
    "What is HNSW in vector databases?",
    "What is the capital of France?",
]

for q in questions:
    registry = build_registry()  # fresh registry so each question has its own log
    result = run_specialist("research", q, registry)

    print("\n" + "=" * 70)
    print("QUESTION:", q)
    print(f"STEPS: {result['steps']}   SUCCESS: {result['success']}")
    print("\nTOOL CALLS:")
    for entry in registry.log:
        print(f"  {entry['tool']}({entry['inputs']})  ->  success={entry['success']}, {entry['latency_ms']} ms")
    print("\nANSWER:")
    print(result["output"])