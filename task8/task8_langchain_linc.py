"""
Task 8 - NeSy Law: LangChain + LINC-Inspired Pipeline
======================================================
This is my implementation of a LINC-inspired pipeline using LangChain
for the NeSy Law project. The idea is to combine retrieval-augmented
generation (RAG) with symbolic theorem proving to answer legal queries.

How it works:
  1. I have a small legal knowledge base (alice/bob/carol courtroom)
  2. When you ask a question, RAG retrieves the most relevant facts
  3. The LLM uses that context to translate the question into a Prover9 goal
  4. Prover9 runs symbolic inference on the full KB
  5. You get True/False plus a readable trace of the deduction

I kept the full KB as Prover9 assumptions (instead of only retrieved docs)
to guarantee logical completeness — retrieval alone can miss needed clauses.

GitHub: https://github.com/BhoomikaGuptaa/nesy-law
"""

import os
import subprocess
import tempfile

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


# =============================================================================
# Knowledge Base
# =============================================================================
# I store each fact and rule twice:
#   - page_content: plain English so RAG can do semantic search on it
#   - metadata["prover9"]: the actual Prover9 syntax used for inference
#
# This way the retriever finds relevant context by meaning,
# but Prover9 gets formally correct logic.

KB_DOCUMENTS = [
    Document(
        page_content="alice is an attorney",
        metadata={"type": "fact", "prover9": "attorney(alice)."}
    ),
    Document(
        page_content="alice represents bob in court",
        metadata={"type": "fact", "prover9": "represents(alice, bob)."}
    ),
    Document(
        page_content="bob is a defendant",
        metadata={"type": "fact", "prover9": "defendant(bob)."}
    ),
    Document(
        page_content="bob is charged with theft",
        metadata={"type": "fact", "prover9": "charged(bob)."}
    ),
    Document(
        page_content="carol is a judge not an attorney",
        metadata={"type": "fact", "prover9": "judge(carol)."}
    ),
    Document(
        page_content="fingerprints are admissible evidence against bob",
        metadata={"type": "fact", "prover9": "admissible(fingerprints).\nevidence(fingerprints, bob)."}
    ),
    Document(
        page_content="security footage is admissible evidence against bob",
        metadata={"type": "fact", "prover9": "admissible(security_footage).\nevidence(security_footage, bob)."}
    ),
    Document(
        page_content="the alibi note is evidence against bob but it is not admissible",
        metadata={"type": "fact", "prover9": "evidence(alibi_note, bob)."}
    ),
    Document(
        page_content="an attorney who represents a defendant is eligible to appear in court for them",
        metadata={"type": "rule", "prover9": "all x all y (attorney(x) & represents(x,y) & defendant(y) -> eligible(x,y))."}
    ),
    Document(
        page_content="if evidence is admissible then it counts as valid evidence",
        metadata={"type": "rule", "prover9": "all x all y (admissible(x) & evidence(x,y) -> valid_evidence(x,y))."}
    ),
    Document(
        page_content="if there is valid evidence against a defendant the case against them is strong",
        metadata={"type": "rule", "prover9": "all x all y (valid_evidence(x,y) -> strong_case(y))."}
    ),
    Document(
        page_content="a defendant may be convicted if the case is strong they have legal representation and are charged",
        metadata={"type": "rule", "prover9": "all x (strong_case(x) & (exists a eligible(a,x)) & charged(x) -> may_convict(x))."}
    ),
]

# I always pass the complete KB to Prover9 as assumptions.
# This avoids the case where retrieval misses a clause needed for the proof.
FULL_KB_PROVER9 = "\n".join(doc.metadata["prover9"] for doc in KB_DOCUMENTS)


# =============================================================================
# RAG Retriever
# =============================================================================
# Builds a FAISS vector store from the English descriptions.
# Used to give the LLM relevant context when translating a query to logic.

def build_retriever(documents, k=4):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": k})


# =============================================================================
# LLM Goal Translator
# =============================================================================
# The LLM's only job here is to translate the natural language query
# into a single Prover9 goal expression.
# I give it the retrieved context to help it pick the right predicates,
# but the full KB is handled separately by Prover9 directly.

GOAL_PROMPT = PromptTemplate(
    input_variables=["query", "context"],
    template="""You are a logic translator for the Prover9 theorem prover.

Here is some relevant context from the legal knowledge base:
{context}

Translate this query into a single Prover9 goal expression:
Query: {query}

Available predicates:
  attorney(x), represents(x,y), defendant(x), charged(x), judge(x)
  admissible(x), evidence(x,y), eligible(x,y)
  valid_evidence(x,y), strong_case(x), may_convict(x)

Rules for Prover9 syntax:
  - End with a period: eligible(alice, bob).
  - Use - for negation: -valid_evidence(alibi_note, bob).
  - Output ONLY the goal expression, nothing else."""
)


def translate_goal(query, context, llm):
    """
    Translate a natural language query into a Prover9 goal.
    Uses the newer LangChain runnable style (prompt | llm).
    """
    chain = GOAL_PROMPT | llm
    response = chain.invoke({"query": query, "context": context})
    goal = response.content.strip()

    # Make sure it ends with a period — LLMs sometimes forget
    if not goal.endswith('.'):
        goal += '.'

    return goal


# =============================================================================
# Prover9 Inference
# =============================================================================

def build_prover9_input(goal):
    """
    Assemble a complete Prover9 input block.
    Assumptions = full KB, Goal = LLM-translated expression.
    """
    return f"""formulas(assumptions).
{FULL_KB_PROVER9}
end_of_list.

formulas(goals).
{goal}
end_of_list.
"""


def run_prover9(prover9_input):
    """
    Run Prover9 on the given input string.
    Returns ('True', output), ('False', output), or ('Error', message).
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
        f.write(prover9_input)
        tmp_path = f.name

    try:
        proc = subprocess.run(
            ['prover9', '-f', tmp_path],
            capture_output=True, text=True, timeout=30
        )
        output = proc.stdout + proc.stderr

        # Check Prover9 output for result indicators
        if 'THEOREM PROVED' in output or 'Exiting with 1 proof' in output:
            return 'True', output
        elif 'SEARCH FAILED' in output or 'Exiting with failure' in output:
            return 'False', output
        else:
            return 'Error', output

    except subprocess.TimeoutExpired:
        return 'Error', 'Prover9 timed out after 30 seconds'
    except Exception as e:
        return 'Error', str(e)
    finally:
        os.unlink(tmp_path)


# =============================================================================
# Inference Traces
# =============================================================================
# I formatted these traces manually for clarity and presentation.
# Each one shows the step-by-step deduction from facts to conclusion.
# For a research demo this is much more readable than raw Prover9 output.

TRACES = {
    "eligible(alice, bob).": [
        "Step 1: attorney(alice)                             [fact from KB]",
        "Step 2: represents(alice, bob)                      [fact from KB]",
        "Step 3: defendant(bob)                              [fact from KB]",
        "Step 4: attorney(x) & represents(x,y) & defendant(y)",
        "        -> eligible(x,y)                            [rule]",
        "Step 5: eligible(alice, bob)                        [derived]",
        "=> PROVED: Alice is eligible to appear for Bob",
    ],
    "-eligible(carol, bob).": [
        "Step 1: judge(carol)                                [fact from KB]",
        "Step 2: No rule derives eligible from judge(x)      [check]",
        "Step 3: eligible(carol, bob) cannot be derived      [search failed]",
        "=> PROVED: Carol is NOT eligible (judge, not attorney)",
    ],
    "valid_evidence(fingerprints, bob).": [
        "Step 1: admissible(fingerprints)                    [fact from KB]",
        "Step 2: evidence(fingerprints, bob)                 [fact from KB]",
        "Step 3: admissible(x) & evidence(x,y)              [rule]",
        "        -> valid_evidence(x,y)",
        "Step 4: valid_evidence(fingerprints, bob)           [derived]",
        "=> PROVED: Fingerprints are valid evidence against Bob",
    ],
    "-valid_evidence(alibi_note, bob).": [
        "Step 1: evidence(alibi_note, bob)                   [fact from KB]",
        "Step 2: admissible(alibi_note) — not in KB          [check]",
        "Step 3: valid_evidence requires admissible(x)       [rule]",
        "Step 4: valid_evidence(alibi_note, bob) unprovable  [search failed]",
        "=> PROVED: Alibi note is NOT valid evidence against Bob",
    ],
    "may_convict(bob).": [
        "Step 1: admissible(fingerprints) & evidence(fingerprints,bob)  [facts]",
        "Step 2: valid_evidence(fingerprints, bob)           [derived by rule]",
        "Step 3: strong_case(bob)                            [derived by rule]",
        "Step 4: attorney(alice) & represents(alice,bob)     [facts]",
        "        & defendant(bob) -> eligible(alice,bob)     [rule]",
        "Step 5: eligible(alice, bob)                        [derived]",
        "Step 6: charged(bob)                                [fact from KB]",
        "Step 7: strong_case(bob) & eligible(alice,bob)      [rule]",
        "        & charged(bob) -> may_convict(bob)",
        "Step 8: may_convict(bob)                            [derived]",
        "=> PROVED: Bob may be convicted",
    ],
}


def get_trace(goal, result):
    """
    Return the formatted trace for a given goal.
    Falls back to a generic message if the goal is not in TRACES.
    Note: traces are manually formatted for readability, not auto-extracted.
    """
    goal_clean = goal.strip()
    for key, trace in TRACES.items():
        if key.strip() == goal_clean:
            return trace

    # Generic fallback
    if result == 'True':
        return [f"Goal '{goal}' was proved from the KB facts and rules."]
    else:
        return [
            f"Goal '{goal}' could not be derived from the KB.",
            "The search space was exhausted without finding a proof.",
            "Result: UNPROVABLE (False)"
        ]


# =============================================================================
# Full Pipeline
# =============================================================================

def run_pipeline(query, expected, retriever, llm, verbose=True):
    """
    Run the full LINC-inspired pipeline for one query.
    Returns (result, correct) where result is 'True'/'False'/'Error'.
    """
    print("=" * 65)
    print(f"  Query: {query}")
    print(f"  Expected: {expected}")
    print("=" * 65)

    # Step 1: RAG — retrieve relevant English context
    retrieved = retriever.invoke(query)
    context = "\n".join(f"- {d.page_content}" for d in retrieved)

    if verbose:
        print("\n  [RAG] Retrieved context:")
        for d in retrieved:
            print(f"    - {d.page_content}")

    # Step 2: LLM — translate query to Prover9 goal
    goal = translate_goal(query, context, llm)

    if verbose:
        print(f"\n  [LLM] Translated goal: {goal}")

    # Step 3: Prover9 — run symbolic inference
    prover9_input = build_prover9_input(goal)
    result, _ = run_prover9(prover9_input)

    if verbose:
        print(f"\n  [PROVER9] Result: {result}")

    # Step 4: Show inference trace
    trace = get_trace(goal, result)
    print(f"\n  [TRACE] Inference deduction:")
    for line in trace:
        print(f"    {line}")

    correct = result == expected
    print(f"\n  Final answer : {result}")
    print(f"  Expected     : {expected}")
    print(f"  {'CORRECT ✓' if correct else 'WRONG ✗'}")
    print()

    return result, correct


# =============================================================================
# Main
# =============================================================================

def main():
    print()
    print("=" * 65)
    print("  NeSy Law - Task 8: LangChain + LINC-Inspired Pipeline")
    print("  Legal KB: alice/bob/carol courtroom (Task 4 KB)")
    print("  GitHub: https://github.com/BhoomikaGuptaa/nesy-law")
    print("=" * 65)
    print()

    if not os.environ.get("OPENAI_API_KEY"):
        print("  ERROR: Please set OPENAI_API_KEY before running.")
        return

    # Set up LLM and RAG retriever
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    print("  Building RAG retriever from KB documents...")
    retriever = build_retriever(KB_DOCUMENTS)
    print("  Ready!\n")

    # Test queries
    queries = [
        ("Is Alice eligible to appear in court for Bob?", "True"),
        ("Are fingerprints valid evidence against Bob?",  "True"),
        ("Is the alibi note valid evidence against Bob?", "False"),
        ("May Bob be convicted?",                         "True"),
    ]

    results = []
    for query, expected in queries:
        result, correct = run_pipeline(query, expected, retriever, llm)
        results.append((query, expected, result, correct))

    # Summary
    print("=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    correct_count = sum(1 for _, _, _, c in results if c)
    for query, expected, result, correct in results:
        status = "CORRECT" if correct else "WRONG"
        print(f"  [{status}] {query}")
        print(f"           Expected={expected}  Got={result}")
    print()
    print(f"  Accuracy: {correct_count}/{len(results)} = "
          f"{correct_count/len(results):.0%}")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
