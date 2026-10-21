import pytest
from reviewer import review_output

EVIDENCE = (
    "[acme_bakery_policy.md] (distance 1.03)\n"
    "Refunds are available within 7 days for unopened packaged goods, but never for custom cakes."
)
CONTEXT = "Output of step 1:\nDelivery is free for orders above 1500 rupees within 10 km of the shop."

# (case name, specialist, output, evidence, task, context, should_be_approved)
CASES = [
    ("good research answer", "research",
     "Refunds are available within 7 days for unopened packaged goods. (Source: acme_bakery_policy.md)",
     EVIDENCE, "Research refund policy", "", True),
    ("wrong number (14 vs 7)", "research",
     "Refunds are available within 14 days for unopened packaged goods.",
     EVIDENCE, "Research refund policy", "", False),
    ("raw JSON from the writer", "writing",
     '{"name": "write_file", "parameters": {"filename": "summary.txt", "content": "Refunds within 7 days"}}',
     "", "Summarize for a new employee", CONTEXT, False),
    ("empty output", "writing", "", "", "Summarize", CONTEXT, False),
    ("research without any search", "research",
     "Refunds are available within 7 days for unopened packaged goods.",
     "", "Research refund policy", "", False),
    ("numbered list is not a fact", "research",
     "1. Refunds are available within 7 days for unopened packaged goods.\n2. Custom cakes are excluded.",
     EVIDENCE, "Research refund policy", "", True),
    ("comma in a number", "writing",
     "Delivery is free for orders above 1,500 rupees within 10 km of the shop.",
     "", "Summarize delivery", CONTEXT, True),
         ("fabricated report without evidence", "research",
     "Market research report: customers want better service. Sources: XYZ Corporation, ABC Inc.",
     "", "Research the market", "", False),
    ("honest no-information answer", "research",
     "The knowledge base has no information on this topic.",
     "", "Research the market", "", True),
]


@pytest.mark.parametrize(
    "specialist,output,evidence,task,context,expected",
    [c[1:] for c in CASES],
    ids=[c[0] for c in CASES],
)
def test_review_decision(specialist, output, evidence, task, context, expected):
    review = review_output(specialist, output, evidence, task, context)
    assert review.approved is expected