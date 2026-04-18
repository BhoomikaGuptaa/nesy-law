"""
Task 6 - NeSy Law: Fuzzy Logic Tutorial
========================================

Fuzzy Logic Tutorial: Applying Fuzzy Reasoning to Legal Scenarios

What is Fuzzy Logic?
---------------------
Classical logic says everything is TRUE (1) or FALSE (0).
Fuzzy logic allows degrees of truth between 0 and 1.

Example: "Is the evidence strong?"
- Classical: Yes or No
- Fuzzy: 0.3 (weak), 0.6 (moderate), 0.9 (strong)

This is especially useful in legal reasoning where concepts like
"reasonable doubt", "excessive force", and "timely manner" have
no sharp boundaries.

Installation:
    pip install scikit-fuzzy numpy matplotlib

GitHub: github.com/BhoomikaGuptaa/nesy-law
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    SKFUZZY_AVAILABLE = True
except ImportError:
    SKFUZZY_AVAILABLE = False
    print("scikit-fuzzy not installed. Run: pip install scikit-fuzzy")
    print("Running simplified demo instead...\n")


# =============================================================================
# PART 1: Manual Fuzzy Logic (no library needed)
# =============================================================================

def triangular_membership(x, a, b, c):
    """
    Triangular membership function.
    Returns degree of membership for value x in triangle (a, b, c).
    a = left foot, b = peak (membership = 1), c = right foot
    """
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)


def trapezoidal_membership(x, a, b, c, d):
    """
    Trapezoidal membership function.
    a = left foot, b = left shoulder, c = right shoulder, d = right foot
    """
    if x <= a or x >= d:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x <= c:
        return 1.0
    else:
        return (d - x) / (d - c)


def demo_manual_fuzzy():
    """
    Demonstrate fuzzy logic manually using membership functions.
    Legal scenario: Is the evidence strong enough to convict?
    """
    print("=" * 65)
    print("  PART 1: Manual Fuzzy Logic — Legal Evidence Scoring")
    print("=" * 65)
    print()
    print("  Scenario: Given an evidence strength score (0 to 1),")
    print("  determine whether the case is weak, moderate, or strong.")
    print()

    # Define membership functions for evidence strength
    # weak: peaks at 0, fades by 0.4
    # moderate: peaks at 0.5, fades by 0.3 on each side
    # strong: starts at 0.6, peaks at 1.0

    test_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    print(f"  {'Evidence Score':<18} {'Weak':>8} {'Moderate':>10} {'Strong':>8}  {'Verdict'}")
    print("  " + "-" * 58)

    for val in test_values:
        weak = triangular_membership(val, 0.0, 0.0, 0.5)
        moderate = triangular_membership(val, 0.2, 0.5, 0.8)
        strong = triangular_membership(val, 0.5, 1.0, 1.0)

        # Pick the category with highest membership
        memberships = {"Weak": weak, "Moderate": moderate, "Strong": strong}
        verdict = max(memberships, key=memberships.get)

        print(f"  {val:<18.1f} {weak:>8.2f} {moderate:>10.2f} {strong:>8.2f}  {verdict}")

    print()


# =============================================================================
# PART 2: Full Fuzzy Inference System with scikit-fuzzy
# =============================================================================

def demo_skfuzzy():
    """
    Build a full Fuzzy Inference System using scikit-fuzzy.
    Legal scenario: Given evidence strength and witness reliability,
    compute a conviction confidence score.
    """
    print("=" * 65)
    print("  PART 2: Full Fuzzy Inference System (scikit-fuzzy)")
    print("=" * 65)
    print()
    print("  Input 1: Evidence strength (0 = none, 1 = overwhelming)")
    print("  Input 2: Witness reliability (0 = unreliable, 1 = very reliable)")
    print("  Output:  Conviction confidence (0 = acquit, 1 = convict)")
    print()

    # ── Define fuzzy variables ────────────────────────────────────────────────
    evidence = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'evidence_strength')
    witness = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'witness_reliability')
    conviction = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'conviction_confidence')

    # ── Membership functions ──────────────────────────────────────────────────
    evidence['weak'] = fuzz.trimf(evidence.universe, [0, 0, 0.5])
    evidence['moderate'] = fuzz.trimf(evidence.universe, [0.2, 0.5, 0.8])
    evidence['strong'] = fuzz.trimf(evidence.universe, [0.5, 1, 1])

    witness['unreliable'] = fuzz.trimf(witness.universe, [0, 0, 0.5])
    witness['reliable'] = fuzz.trimf(witness.universe, [0.5, 1, 1])

    conviction['acquit'] = fuzz.trimf(conviction.universe, [0, 0, 0.4])
    conviction['uncertain'] = fuzz.trimf(conviction.universe, [0.3, 0.5, 0.7])
    conviction['convict'] = fuzz.trimf(conviction.universe, [0.6, 1, 1])

    # ── Fuzzy rules ───────────────────────────────────────────────────────────
    # Rule 1: Strong evidence AND reliable witness → convict
    rule1 = ctrl.Rule(evidence['strong'] & witness['reliable'], conviction['convict'])
    # Rule 2: Weak evidence → acquit
    rule2 = ctrl.Rule(evidence['weak'], conviction['acquit'])
    # Rule 3: Moderate evidence AND unreliable witness → uncertain
    rule3 = ctrl.Rule(evidence['moderate'] & witness['unreliable'], conviction['uncertain'])
    # Rule 4: Moderate evidence AND reliable witness → convict
    rule4 = ctrl.Rule(evidence['moderate'] & witness['reliable'], conviction['convict'])
    # Rule 5: Strong evidence AND unreliable witness → uncertain
    rule5 = ctrl.Rule(evidence['strong'] & witness['unreliable'], conviction['uncertain'])

    # ── Build and run the system ──────────────────────────────────────────────
    conviction_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
    sim = ctrl.ControlSystemSimulation(conviction_ctrl)

    # ── Test scenarios ────────────────────────────────────────────────────────
    scenarios = [
        (0.1, 0.3, "Weak evidence, unreliable witness"),
        (0.5, 0.5, "Moderate evidence, moderate witness"),
        (0.5, 0.9, "Moderate evidence, reliable witness"),
        (0.9, 0.3, "Strong evidence, unreliable witness"),
        (0.9, 0.9, "Strong evidence, reliable witness"),
    ]

    print(f"  {'Scenario':<42} {'Evidence':>9} {'Witness':>8} {'Confidence':>11}  {'Decision'}")
    print("  " + "-" * 82)

    for ev, wit, desc in scenarios:
        sim.input['evidence_strength'] = ev
        sim.input['witness_reliability'] = wit
        sim.compute()
        conf = sim.output['conviction_confidence']

        if conf < 0.35:
            decision = "Acquit"
        elif conf < 0.65:
            decision = "Uncertain"
        else:
            decision = "Convict"

        print(f"  {desc:<42} {ev:>9.1f} {wit:>8.1f} {conf:>11.2f}  {decision}")

    print()


# =============================================================================
# PART 3: LLM + Fuzzy Logic Integration Concept
# =============================================================================

def demo_llm_fuzzy_concept():
    """
    Demonstrate how LLM output probabilities can be used as
    fuzzy membership values. This simulates what an LLM would output
    when asked to assess legal scenarios.
    """
    print("=" * 65)
    print("  PART 3: LLM + Fuzzy Logic Integration")
    print("=" * 65)
    print()
    print("  Concept: LLM confidence scores can serve as fuzzy")
    print("  membership values, combining neural and symbolic reasoning.")
    print()

    # Simulated LLM outputs for legal questions
    llm_outputs = [
        {
            "question": "Is the alibi note admissible as evidence?",
            "llm_confidence": 0.15,
            "note": "LLM thinks it's probably not admissible"
        },
        {
            "question": "Did the defendant act with reasonable force?",
            "llm_confidence": 0.55,
            "note": "LLM is uncertain — grey area"
        },
        {
            "question": "Is Alice eligible to represent Bob in court?",
            "llm_confidence": 0.92,
            "note": "LLM is highly confident — yes"
        },
    ]

    for item in llm_outputs:
        conf = item["llm_confidence"]

        # Use LLM confidence as fuzzy membership value
        not_likely = triangular_membership(conf, 0.0, 0.0, 0.5)
        uncertain = triangular_membership(conf, 0.2, 0.5, 0.8)
        likely = triangular_membership(conf, 0.5, 1.0, 1.0)

        print(f"  Q: {item['question']}")
        print(f"  LLM confidence: {conf}")
        print(f"  Note: {item['note']}")
        print(f"  Fuzzy interpretation:")
        print(f"    Not likely:  {not_likely:.2f}")
        print(f"    Uncertain:   {uncertain:.2f}")
        print(f"    Likely:      {likely:.2f}")
        print()


# =============================================================================
# PART 4: Plot membership functions
# =============================================================================

def plot_membership_functions():
    """Plot the fuzzy membership functions for evidence strength."""
    x = np.arange(0, 1.1, 0.01)

    weak = [triangular_membership(v, 0.0, 0.0, 0.5) for v in x]
    moderate = [triangular_membership(v, 0.2, 0.5, 0.8) for v in x]
    strong = [triangular_membership(v, 0.5, 1.0, 1.0) for v in x]

    plt.figure(figsize=(8, 4))
    plt.plot(x, weak, 'b-', label='Weak', linewidth=2)
    plt.plot(x, moderate, 'g-', label='Moderate', linewidth=2)
    plt.plot(x, strong, 'r-', label='Strong', linewidth=2)
    plt.xlabel('Evidence Strength')
    plt.ylabel('Degree of Membership')
    plt.title('Fuzzy Membership Functions for Evidence Strength')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fuzzy_membership_functions.png', dpi=150)
    plt.show()
    print("  Plot saved as fuzzy_membership_functions.png")
    print()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print()
    print("=" * 65)
    print("  NeSy Law — Task 6: Fuzzy Logic Tutorial")
    print("  github.com/BhoomikaGuptaa/nesy-law")
    print("=" * 65)
    print()

    # Part 1 always runs (no library needed)
    demo_manual_fuzzy()

    # Part 2 requires scikit-fuzzy
    if SKFUZZY_AVAILABLE:
        demo_skfuzzy()
    else:
        print("  Skipping Part 2 — install scikit-fuzzy to run full demo")
        print()

    # Part 3 always runs
    demo_llm_fuzzy_concept()

    # Part 4 — plot
    try:
        plot_membership_functions()
    except Exception as e:
        print(f"  Could not generate plot: {e}")
        print()

    print("=" * 65)
    print("  Tutorial complete!")
    print("  Install scikit-fuzzy for the full inference system:")
    print("  pip install scikit-fuzzy")
    print("=" * 65)
    print()
