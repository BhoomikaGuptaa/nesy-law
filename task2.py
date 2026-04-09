

import os
import subprocess
import tempfile
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# ── All scenarios ─────────────────────────────────────────────────────────────
scenarios = [
    {
        "tag": "REFERENCE EXAMPLE",
        "name": "Algorithm Characteristics",
        "description": "Translating CS knowledge into Prolog facts and rules",
        "input": "DFS is memory efficient. BFS finds the shortest path. An algorithm is preferred for deep search if it is memory efficient.",
        "question": "Is DFS preferred for deep search?"
    },
    {
        "tag": "LEGAL SCENARIO 1",
        "name": "Establishing the Parties",
        "description": "Who is involved in the case?",
        "input": "Alice is a licensed attorney. Bob is Alice's client. Carol is a judge. Dan is a witness.",
        "question": "Is Alice a licensed attorney?"
    },
    {
        "tag": "LEGAL SCENARIO 2",
        "name": "The Contract",
        "description": "What happened between the parties?",
        "input": "Alice and Bob signed a contract on January 15 2024. The contract involves a property dispute. Bob paid Alice a retainer fee.",
        "question": "Did Alice and Bob sign a contract?"
    },
    {
        "tag": "LEGAL SCENARIO 3",
        "name": "Case Eligibility",
        "description": "Is this case eligible for court?",
        "input": "A case is eligible for court if it involves a licensed attorney and a signed contract. Alice is a licensed attorney. Alice and Bob have a signed contract.",
        "question": "Is the case between Alice and Bob eligible for court?"
    }
]

SYSTEM_PROMPT = """You are a translator from English to SWI-Prolog.
For every fact in the user input, output it as a Prolog fact or rule.
Rules:
- Output ONLY Prolog code, no explanation, no markdown, no backticks
- All atom names must be lowercase (alice, bob, dfs, bfs)
- Variables in rules must be uppercase (X, Y, Z)
- End every fact and rule with a period
- Keep it minimal, only output what is stated
- Write a main/0 predicate using -> ; pattern to answer the question
- End with: :- initialization(main, main).

Example:
Input: Alice is an attorney. Bob is Alice's client.
Question: Is Alice an attorney?
Output:
attorney(alice).
client(bob, alice).
main :- (attorney(alice) -> writeln('true') ; writeln('false')).
:- initialization(main, main)."""


def translate_and_query(text, question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Statement: {text}\nQuestion: {question}"}
        ],
        temperature=0
    )
    code = response.choices[0].message.content.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        code = "\n".join(lines[1:-1])
    return code


def run_prolog(code):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pl", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            ["swipl", "-q", tmp_path],
            capture_output=True, text=True, timeout=15,
            encoding="utf-8", errors="replace"
        )
        output = result.stdout.strip()
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "[ERROR] timed out"
    except FileNotFoundError:
        return "[ERROR] SWI-Prolog not found"
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


# ── Run pipeline ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  NeSy Law — Task 2: Natural Language to Prolog Pipeline")
print("=" * 65)
print("\n  Approach: LLM extracts Prolog facts from natural language,")
print("  SWI-Prolog runs backward chaining to answer queries.\n")

for i, scenario in enumerate(scenarios, 1):
    print(f"{'=' * 65}")
    print(f"  [{scenario['tag']}]  {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"{'=' * 65}")
    print(f"\n  Input:    \"{scenario['input']}\"")
    print(f"  Question: \"{scenario['question']}\"")

    prolog = translate_and_query(scenario["input"], scenario["question"])

    print(f"\n  Generated Prolog:")
    for line in prolog.split("\n"):
        if line.strip():
            print(f"    {line}")

    result = run_prolog(prolog)
    verdict = "true" if result == "true" else "false" if result == "false" else result

    print(f"\n  Result: {verdict}\n")

print("=" * 65)
print("  Pipeline complete.")
print("=" * 65 + "\n")