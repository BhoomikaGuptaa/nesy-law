"""
Task 4 - NeSy Law: Build Your Own Knowledge Base

Knowledge Base Theme: A Legal Dispute
Inspired by the Simpsons KB example from the course reference material,
this KB models a small courtroom world with people, roles, evidence,
and legal rules — connecting symbolic reasoning to the NeSy Law project.

The KB has 14 facts and 3 rules. Queries are run using janus-swi,
SWI-Prolog's official Python binding, which allows Prolog to run
directly inside Python without subprocess calls.

Local Setup:
- SWI-Prolog 10.0.2 installed on Windows, added to system PATH
- Python 3.14
- janus-swi installed via: pip install janus-swi
"""

import janus_swi as janus
import tempfile
import os

# ── Knowledge Base ────────────────────────────────────────────────────────────
KNOWLEDGE_BASE = """
% --- FACTS: Roles (6 facts) ---
attorney(alice).
attorney(carlos).
judge(carol).
witness(dan).
witness(eve).
defendant(bob).

% --- FACTS: Representation (2 facts) ---
represents(alice, bob).
represents(carlos, state).

% --- FACTS: Evidence (3 facts) ---
evidence(fingerprints, bob).
evidence(security_footage, bob).
evidence(alibi_note, bob).

% --- FACTS: Admissibility (2 facts) ---
% alibi_note is intentionally not marked admissible to test rule rejection
admissible(fingerprints).
admissible(security_footage).

% --- FACTS: Charges (1 fact) ---
charged(bob, theft).

% --- RULES ---

% Rule 1: Evidence is valid if it is admissible and links to the defendant.
valid_evidence(E, D) :-
    evidence(E, D),
    admissible(E).

% Rule 2: A case is strong if there are two distinct pieces of valid evidence.
strong_case(D) :-
    valid_evidence(E1, D),
    valid_evidence(E2, D),
    E1 \\= E2.

% Rule 3: A defendant may be convicted if the case is strong,
% they have legal representation, and they are charged.
may_be_convicted(D) :-
    strong_case(D),
    represents(_, D),
    charged(D, _).
"""

# Load KB by writing to a temp file and consulting it
with tempfile.NamedTemporaryFile(mode="w", suffix=".pl", delete=False, encoding="utf-8") as f:
    f.write(KNOWLEDGE_BASE)
    kb_path = f.name.replace("\\", "/")

janus.query_once(f"consult('{kb_path}')")

# ── Helper functions ──────────────────────────────────────────────────────────
def run_query(goal):
    """Run a yes/no query using janus directly inside Python."""
    result = janus.query_once(goal)
    return "true" if result.get("truth") else "false"

def run_findall(var, goal):
    """Run a findall query and return all results as a list."""
    results = [str(sol[var]) for sol in janus.query(goal)]
    return results if results else ["none"]

# ── Queries ───────────────────────────────────────────────────────────────────
yes_no_queries = [
    ("attorney(alice)",                      "Is Alice an attorney?"),
    ("judge(carol)",                         "Is Carol the judge?"),
    ("witness(dan)",                         "Is Dan a witness?"),
    ("admissible(fingerprints)",             "Are fingerprints admissible?"),
    ("admissible(alibi_note)",               "Is the alibi note admissible?"),
    ("valid_evidence(fingerprints, bob)",    "Are fingerprints valid evidence against Bob?"),
    ("valid_evidence(alibi_note, bob)",      "Is the alibi note valid evidence against Bob?"),
    ("strong_case(bob)",                     "Is the case against Bob strong?"),
    ("may_be_convicted(bob)",                "May Bob be convicted?"),
    ("represents(alice, bob)",               "Does Alice represent Bob?"),
]

list_queries = [
    ("E", "valid_evidence(E, bob)",  "What is the valid evidence against Bob?"),
    ("X", "attorney(X)",             "Who are the attorneys?"),
    ("X", "witness(X)",              "Who are the witnesses?"),
    ("X", "charged(X, theft)",       "Who is charged with theft?"),
]

# ── Output ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  NeSy Law — Task 4: Legal Dispute Knowledge Base")
print("=" * 65)
print("\n  14 facts · 3 rules · courtroom domain")
print("  Using janus-swi: SWI-Prolog running directly inside Python\n")

print("  YES / NO QUERIES")
print("  " + "-" * 58)
for goal, desc in yes_no_queries:
    result = run_query(goal)
    print(f"  {desc:<48} {result}")

print("\n  LIST QUERIES")
print("  " + "-" * 58)
for var, goal, desc in list_queries:
    results = run_findall(var, goal)
    print(f"  {desc}")
    for r in results:
        print(f"    > {r}")
    print()

print("=" * 65)
print("  All queries complete.")
print("=" * 65 + "\n")

# Cleanup temp file
try:
    os.unlink(kb_path)
except:
    pass
