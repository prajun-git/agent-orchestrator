from agent import run_specialist
from registry import ToolRegistry
from reviewer import review_output
from schemas import Plan
from working_memory import WorkingMemory

MAX_ATTEMPTS = 2  # first try plus one revision


def _run_with_review(specialist: str, task: str, registry: ToolRegistry, context: str) -> dict:
    feedback = ""
    last = {"success": False, "output": "not run", "steps": 0, "evidence": ""}

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            last = run_specialist(specialist, task, registry, context, feedback)
        except Exception as e:  # e.g. quota or network errors
            last = {"success": False, "output": f"{type(e).__name__}: {e}", "steps": 0, "evidence": ""}

        if not last["success"]:
            print(f"    attempt {attempt} failed: {str(last['output'])[:100]}")
            continue

        review = review_output(specialist, last["output"], last["evidence"], task, context)
        if review.approved:
            return {"status": "done", "output": last["output"], "steps": last["steps"]}

        print(f"    attempt {attempt} rejected by reviewer:")
        for issue in review.issues:
            print(f"      - {issue}")
        feedback = "\n".join(f"- {i}" for i in review.issues)

    status = "rejected" if last["success"] else "failed"
    return {"status": status, "output": last["output"], "steps": last["steps"]}


def execute_plan(plan: Plan, registry: ToolRegistry, memory: WorkingMemory | None = None) -> dict[int, dict]:
    memory = memory or WorkingMemory()
    memory.set("plan", plan.model_dump())

    results: dict[int, dict] = {}
    pending = {s.id: s for s in plan.subtasks}

    while pending:
        ready = [s for s in pending.values() if all(d in results for d in s.depends_on)]

        if not ready:  # nothing can ever run: bad ids or a dependency cycle
            for s in pending.values():
                results[s.id] = {"status": "failed", "output": "unresolvable dependencies", "steps": 0}
                memory.record_step(s.id, "failed", "unresolvable dependencies")
                memory.log_error(s.id, "unresolvable dependencies")
            break

        for s in sorted(ready, key=lambda x: x.id):
            del pending[s.id]

            bad_deps = [d for d in s.depends_on if results[d]["status"] != "done"]
            if bad_deps:
                print(f"  [executor] step {s.id} skipped (depends on unfinished step {bad_deps})")
                msg = f"skipped: step(s) {bad_deps} did not finish"
                results[s.id] = {"status": "skipped", "output": msg, "steps": 0}
                memory.record_step(s.id, "skipped", msg)
                memory.log_error(s.id, msg)
                continue

            # Outputs of earlier steps now come from working memory, not a private dict.
            context = "\n\n".join(f"Output of step {d}:\n{memory.step_output(d)}" for d in s.depends_on)
            task = f"{s.description}\nExpected output: {s.expected_output}"
            print(f"  [executor] step {s.id} ({s.specialist}): {s.description}")

            results[s.id] = _run_with_review(s.specialist, task, registry, context)
            memory.record_step(s.id, results[s.id]["status"], results[s.id]["output"])
            if results[s.id]["status"] != "done":
                memory.log_error(s.id, f"step ended as {results[s.id]['status']}")
    return results