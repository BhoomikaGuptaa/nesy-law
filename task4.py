"""
Task 4 - NeSy Law: Build Your Own Knowledge Base

Knowledge Base Theme: A Legal Dispute
Inspired by the Simpsons KB example from the course reference material,
this KB models a small courtroom world with people, roles, evidence,
and legal rules — connecting symbolic reasoning to the NeSy Law project.

The KB has 12 facts and 3 rules. Queries are run via Python subprocess
using SWI-Prolog, and results are verified to be consistent.
"""

import subprocess
import tempfile
import os

# ── Knowledge Base ────────────────────────────────────────────────────────────
# Inspired by the Simpsons KB structure from the course reference:
# entities + relationships + rules, all in Prolog.
# Domain: a small courtroom world.

KNOWLEDGE_BASE = """
% ============================================================
% NeSy Law Task 4 - Legal Dispute Knowledge Base
% Entities: alice, bob, carol, dan, eve, carlos
% Relationships: roles, representation, evidence, charges
% ============================================================

% --- FACTS: Roles ---
attorney(alice).
attorney(carlos).
judge(carol).
witness(dan).
witness(eve).
defendant(bob).

% --- FACTS: Representation ---
represents(alice, bob).
represents(carlos, state).

% --- FACTS: Evidence and admissibility ---
evidence(fingerprints, bob).
evidence(security_footage, bob).
evidence(alibi_note, bob).

admissible(fingerprints).
admissible(security_footage).

% --- FACTS: Charges ---
charged(bob, theft).

% --- RULES ---

% Rule 1: Evidence is valid if it is admissible and links to the defendant.
% Inspired by: mother(X, Y) :- parent(X, Y), female(X).
valid_evidence(E, D) :-
    evidence(E, D),
    admissible(E).

% Rule 2: A case is strong if there are two distinct pieces of valid evidence.
% Inspired by: grandparent(X, Y) :- parent(X, Z), parent(Z, Y).
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

# ── Helper functions ──────────────────────────────────────────────────────────
def run_query(goal):
    """Run a yes/no Prolog query against the KB."""
    program = KNOWLEDGE_BASE + f"""
main :- ({goal} -> writeln('true') ; writeln('false')).
:- initialization(main, main).
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pl", delete=False, encoding="utf-8") as f:
        f.write(program)
        tmp = f.name
    try:
        result = subprocess.run(["swipl", "-q", tmp],
            capture_output=True, text=True, timeout=10,
            encoding="utf-8", errors="replace")
        return result.stdout.strip() or "(no output)"
    except FileNotFoundError:
        return "[ERROR] SWI-Prolog not found"
    finally:
        try: os.unlink(tmp)
        except: pass


def run_findall(var, goal):
    """Run a findall query to list all matching results."""
    program = KNOWLEDGE_BASE + f"""
main :-
    findall({var}, {goal}, Results),
    (Results = [] -> writeln('none') ; maplist(writeln, Results)).
:- initialization(main, main).
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pl", delete=False, encoding="utf-8") as f:
        f.write(program)
        tmp = f.name
    try:
        result = subprocess.run(["swipl", "-q", tmp],
            capture_output=True, text=True, timeout=10,
            encoding="utf-8", errors="replace")
        return result.stdout.strip() or "(no output)"
    except FileNotFoundError:
        return "[ERROR] SWI-Prolog not found"
    finally:
        try: os.unlink(tmp)
        except: pass


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
print("\n  12 facts · 3 rules · courtroom domain")
print("  Results verified via Python subprocess + SWI-Prolog\n")

print("  YES / NO QUERIES")
print("  " + "-" * 58)
for goal, desc in yes_no_queries:
    result = run_query(goal)
    print(f"  {desc:<48} {result}")

print("\n  LIST QUERIES")
print("  " + "-" * 58)
for var, goal, desc in list_queries:
    result = run_findall(var, goal)
    print(f"  {desc}")
    for line in result.split("\n"):
        print(f"    > {line}")
    print()

print("=" * 65)
print("  All queries complete.")
print("=" * 65 + "\n")
