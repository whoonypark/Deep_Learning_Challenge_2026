"""Unit tests for answer extraction. Run: python tests/test_answer_extraction.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from answer_extraction import extract_boxed_int, extract_final_int, majority_vote

CASES = [
    # (model output, expected)
    ("... so the answer is \\boxed{42}.", 42),
    ("Thus \\boxed{-17} is the result.", -17),
    ("We get \\boxed{1{,}234}", 1234),
    ("We get \\boxed{1,234,567}", 1234567),
    ("So \\boxed{3431577212128939}", 3431577212128939),
    ("\\boxed{42.0}", 42),
    ("\\boxed{\\frac{84}{2}}", 42),
    ("\\boxed{\\dfrac{-84}{2}}", -42),
    ("\\boxed{ 650 }", 650),
    ("\\boxed{650\\text{ miles}}", 650),
    ("\\boxed{$1200$}", 1200),
    ("first \\boxed{1} then finally \\boxed{2}", 2),
    ("nested \\boxed{\\left( 99 \\right)}", 99),
    ("The final answer is 185.", 185),
    ("Answer: 5000", 5000),
    ("The answer is $8$ pounds... wait, recompute: 9", 8),  # phrase beats last int
    ("no numbers here at all", None),
    ("", None),
    ("The result is 1,234 apples.", 1234),
    ("\\boxed 7", 7),
]


def main() -> None:
    failed = 0
    for text, want in CASES:
        got = extract_final_int(text)
        if got != want:
            failed += 1
            print(f"FAIL extract_final_int({text!r}) = {got!r}, want {want!r}")

    # strict boxed variant
    strict_cases = [("the answer is 42", None), ("\\boxed{42}", 42)]
    for text, want in strict_cases:
        got = extract_boxed_int(text)
        if got != want:
            failed += 1
            print(f"FAIL extract_boxed_int({text!r}) = {got!r}, want {want!r}")

    votes = [
        ([1, 2, 2, None], 2),
        ([None, None], None),
        ([5], 5),
        ([3, 3, 7, 7, 1], 3),  # tie -> first seen
    ]
    for answers, want in votes:
        got = majority_vote(answers)
        if got != want:
            failed += 1
            print(f"FAIL majority_vote({answers}) = {got!r}, want {want!r}")

    total = len(CASES) + len(strict_cases) + len(votes)
    print(f"{total - failed}/{total} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
