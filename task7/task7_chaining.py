"""
Task 7 - NeSy Law: Forward and Backward Chaining
=================================================
Assignment 4, Problems 4, 5, and 6.

Uses the MIT 6.034 production system library:
    production.py  - IF, AND, OR, NOT, THEN, forward_chain, match, populate, simplify
    data.py        - poker_data, simpsons_data, black_data, zookeeper_rules
    utils.py       - helper utilities used by production.py

GitHub: https://github.com/BhoomikaGuptaa/nesy-law
"""

from production import IF, AND, OR, NOT, THEN, forward_chain, match, populate, simplify
from data import poker_data, simpsons_data, black_data, zookeeper_rules


# =============================================================================
# PROBLEM 4 - Transitive Rule
# =============================================================================

def transitive_rule():
    """
    A single forward-chaining rule for transitivity:
    if X beats Z and Z beats Y, then X beats Y.
    """
    return IF(
        AND('(?x) beats (?z)',
            '(?z) beats (?y)'),
        THEN('(?x) beats (?y)')
    )


# =============================================================================
# PROBLEM 5 - Family Relations
# =============================================================================

def family_rules():
    """
    Forward-chaining rules for deriving family relations from:
        person (?x)
        parent (?x) (?y)

    Derived relations:
        sibling, child, grandparent, grandchild, cousin
    """
    return [
        # Helper relation used to prevent self-sibling and self-cousin matches
        IF('person (?x)',
           THEN('self (?x) (?x)')),

        # Two distinct people with a shared parent are siblings
        IF(AND('parent (?p) (?x)',
               'parent (?p) (?y)',
               NOT('self (?x) (?y)')),
           THEN('sibling (?x) (?y)')),

        # Reverse parent relation
        IF('parent (?x) (?y)',
           THEN('child (?y) (?x)')),

        # Parent of a parent
        IF(AND('parent (?gp) (?p)',
               'parent (?p) (?x)'),
           THEN('grandparent (?gp) (?x)')),

        # Reverse grandparent relation
        IF('grandparent (?gp) (?x)',
           THEN('grandchild (?x) (?gp)')),

        # Children of siblings are cousins, excluding siblings and self-matches
        IF(AND('sibling (?p1) (?p2)',
               'parent (?p1) (?x)',
               'parent (?p2) (?y)',
               NOT('self (?x) (?y)'),
               NOT('sibling (?x) (?y)')),
           THEN('cousin (?x) (?y)')),
    ]


# =============================================================================
# PROBLEM 6 - Backward Chaining Goal Tree
# =============================================================================

def backchain_to_goal_tree(rules, hypothesis):
    """
    Build an AND/OR goal tree showing what must be true
    in order to prove the given hypothesis.

    Leaves are strings representing atomic facts.
    Internal nodes are AND / OR expressions.
    """

    def expand(antecedent, bindings):
        """
        Recursively populate variables in an antecedent and backchain on it.
        Handles strings, AND expressions, and OR expressions uniformly.
        """
        if isinstance(antecedent, str):
            return backchain_to_goal_tree(rules, populate(antecedent, bindings))
        elif isinstance(antecedent, AND):
            return AND(*[expand(part, bindings) for part in antecedent])
        elif isinstance(antecedent, OR):
            return OR(*[expand(part, bindings) for part in antecedent])
        return antecedent

    branches = [hypothesis]

    for rule in rules:
        consequent = rule.consequent()
        bindings = match(consequent, hypothesis)
        if bindings is not None:
            branches.append(expand(rule.antecedent(), bindings))

    return simplify(OR(*branches))


# =============================================================================
# Demos
# =============================================================================

def demo_problem4():
    result = forward_chain([transitive_rule()], poker_data)
    derived = [fact for fact in result if fact not in poker_data]

    print("=" * 55)
    print("  Problem 4: Transitive Poker Rankings")
    print("=" * 55)
    print(f"  Original facts : {len(poker_data)}")
    print(f"  Derived facts  : {len(derived)}")
    print(f"  Total facts    : {len(result)}")
    print()
    print("  Sample derived rankings:")
    for r in sorted(derived)[:5]:
        print(f"    {r}")
    print(f"    ... and {len(derived) - 5} more")
    print()


def demo_problem5():
    print("=" * 55)
    print("  Problem 5: Family Relations")
    print("=" * 55)

    simpsons_result = forward_chain(family_rules(), simpsons_data)
    print("  Simpsons family:")
    for relation in ['sibling', 'child', 'grandparent', 'grandchild', 'cousin']:
        matches = [r for r in simpsons_result if r.startswith(relation)]
        print(f"    {relation}: {len(matches)}")
    print()

    black_result = forward_chain(family_rules(), black_data)
    black_cousins = [r for r in black_result if r.startswith('cousin')]
    print("  Harry Potter Black family:")
    for relation in ['sibling', 'child', 'grandparent', 'grandchild', 'cousin']:
        matches = [r for r in black_result if r.startswith(relation)]
        print(f"    {relation}: {len(matches)}")
    print(f"  Expected 14 cousins: {len(black_cousins) == 14}")
    print()


def demo_problem6():
    print("=" * 55)
    print("  Problem 6: Backward Chaining Goal Trees")
    print("=" * 55)
    print()
    print("  'opus is a penguin':")
    print(" ", backchain_to_goal_tree(zookeeper_rules, 'opus is a penguin'))
    print()
    print("  'opus is a mammal':")
    print(" ", backchain_to_goal_tree(zookeeper_rules, 'opus is a mammal'))
    print()


if __name__ == "__main__":
    demo_problem4()
    demo_problem5()
    demo_problem6()
