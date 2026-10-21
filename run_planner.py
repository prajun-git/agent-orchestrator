from planner import create_plan

tasks = [
    "Research the top 3 vector databases, compare them, and write a short summary for a CTO.",
    "Write a summary of the pros and cons of microservices.",
    "Make my business better.",
]

for task in tasks:
    plan = create_plan(task)
    print(f"\nTASK: {task}")
    print(f"GOAL: {plan.goal}  (confidence: {plan.confidence})")
    print(f"ASSUMPTIONS: {plan.assumptions}")
    print(f"BLOCKING: {plan.blocking_questions}")
    for s in plan.subtasks:
        print(f"  {s.id}. [{s.specialist}] {s.description}  depends_on={s.depends_on}  ({s.complexity})")