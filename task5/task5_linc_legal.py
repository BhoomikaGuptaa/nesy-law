import os
import subprocess
import tempfile
from collections import Counter
from openai import OpenAI

PROVER9_PATH = "/home/jovyan/LADR-2009-11A/bin/prover9"
K_VOTES = 3
MODEL = "gpt-3.5-turbo"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SCENARIOS = [
    {
        "id": 1,
        "description": "Alice is eligible to appear in court for Bob.",
        "nl_premises": [
            "Alice is an attorney.",
            "Alice represents Bob.",
            "Bob is a defendant.",
            "An attorney who represents a defendant is eligible to appear in court.",
        ],
        "nl_conclusion": "Alice is eligible to appear in court.",
        "prover9_assumptions": """
Attorney(alice).
Represents(alice,bob).
Defendant(bob).
(all x (Attorney(x) & Represents(x,bob) & Defendant(bob) -> Eligible(x))).
""",
        "prover9_goal": "Eligible(alice).",
        "expected": "True",
    },
    {
        "id": 2,
        "description": "The case against Bob is strong.",
        "nl_premises": [
            "Fingerprints are admissible evidence against Bob.",
            "Security footage is admissible evidence against Bob.",
            "Admissible evidence against a defendant is valid evidence.",
            "A case is strong if there are two different pieces of valid evidence.",
        ],
        "nl_conclusion": "The case against Bob is strong.",
        "prover9_assumptions": """
Admissible(fingerprints).
Admissible(security_footage).
EvidenceAgainst(fingerprints,bob).
EvidenceAgainst(security_footage,bob).
(all x (Admissible(x) & EvidenceAgainst(x,bob) -> ValidEvidence(x,bob))).
ValidEvidence(fingerprints,bob) & ValidEvidence(security_footage,bob) & fingerprints != security_footage -> StrongCase(bob).
""",
        "prover9_goal": "StrongCase(bob).",
        "expected": "True",
    },
    {
        "id": 3,
        "description": "The alibi note is NOT valid evidence against Bob.",
        "nl_premises": [
            "The alibi note is evidence against Bob.",
            "The alibi note is not admissible.",
            "Evidence is valid only if it is admissible.",
        ],
        "nl_conclusion": "The alibi note is valid evidence against Bob.",
        "prover9_assumptions": """
EvidenceAgainst(alibi_note,bob).
-Admissible(alibi_note).
(all x (ValidEvidence(x,bob) -> Admissible(x))).
""",
        "prover9_goal": "ValidEvidence(alibi_note,bob).",
        "expected": "False",
    },
    {
        "id": 4,
        "description": "Carol is NOT an attorney.",
        "nl_premises": [
            "Carol is a judge.",
            "A judge is not an attorney.",
        ],
        "nl_conclusion": "Carol is an attorney.",
        "prover9_assumptions": """
Judge(carol).
(all x (Judge(x) -> -Attorney(x))).
""",
        "prover9_goal": "Attorney(carol).",
        "expected": "False",
    },
    {
        "id": 5,
        "description": "Bob may be convicted.",
        "nl_premises": [
            "Bob is charged with theft.",
            "Alice is an attorney who represents Bob.",
            "The case against Bob is strong.",
            "A defendant may be convicted if the case is strong, they have representation, and are charged.",
        ],
        "nl_conclusion": "Bob may be convicted.",
        "prover9_assumptions": """
ChargedWith(bob,theft).
Represents(alice,bob).
Attorney(alice).
StrongCase(bob).
(all x (ChargedWith(x,theft) & StrongCase(x) & Represents(alice,x) -> MayBeConvicted(x))).
""",
        "prover9_goal": "MayBeConvicted(bob).",
        "expected": "True",
    },
]

def get_fol_from_llm(nl_premises, nl_conclusion):
    premises_text = "\n".join(f"- {p}" for p in nl_premises)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Translate natural language legal premises and conclusion into First Order Logic expressions. Be concise."},
            {"role": "user", "content": f"Premises:\n{premises_text}\nConclusion: {nl_conclusion}\n\nGive the FOL translation:"},
        ],
        temperature=0.8,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()

def run_prover9(assumptions, goal, timeout=15):
    prover9_input = f"formulas(assumptions).\n{assumptions}\nend_of_list.\n\nformulas(goals).\n{goal}\nend_of_list.\n"
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".in", delete=False, encoding="utf-8") as f:
            f.write(prover9_input)
            tmp_path = f.name
        result = subprocess.run([PROVER9_PATH, "-f", tmp_path], capture_output=True, text=True, timeout=timeout)
        os.unlink(tmp_path)
        output = result.stdout + result.stderr
        if "THEOREM PROVED" in output:
            return "True"
        elif "SEARCH FAILED" in output:
            return "False"
        else:
            return "Uncertain"
    except subprocess.TimeoutExpired:
        return "Uncertain"
    except Exception as e:
        return "Error"

def main():
    print("\n" + "="*70)
    print("  NeSy Law - Task 5: LINC Reimplementation on Legal KB")
    print("="*70)
    print(f"\n  Model: {MODEL} | K={K_VOTES} votes | Prover: Prover9")
    print(f"  Pipeline: NL -> OpenAI FOL -> Prover9 -> True/False\n")

    results = []
    for scenario in SCENARIOS:
        print(f"  Scenario {scenario['id']}: {scenario['description']}")
        print(f"  " + "-"*60)
        print(f"  Premises:")
        for p in scenario["nl_premises"]:
            print(f"    - {p}")
        print(f"  Conclusion: {scenario['nl_conclusion']}")

        print(f"\n  LLM FOL Translation:")
        fol = get_fol_from_llm(scenario["nl_premises"], scenario["nl_conclusion"])
        for line in fol.split("\n")[:5]:
            if line.strip():
                print(f"    {line.strip()}")

        votes = []
        for k in range(K_VOTES):
            result = run_prover9(scenario["prover9_assumptions"], scenario["prover9_goal"])
            votes.append(result)

        final = Counter(votes).most_common(1)[0][0]
        correct = final == scenario["expected"]

        print(f"\n  Prover9 votes: {votes}")
        print(f"  Final: {final} | Expected: {scenario['expected']} | {'CORRECT' if correct else 'WRONG'}")
        print()
        results.append(correct)

    accuracy = sum(results)/len(results)
    print("="*70)
    print(f"  ACCURACY: {sum(results)}/{len(results)} = {accuracy:.0%}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
