"""Robust extraction of the final integer answer from a model response.

Priority:
  1. last \\boxed{...} content
  2. "answer is X" style phrases
  3. last standalone integer in the text

All answers in this challenge are integers (can be negative and larger than
1e15, so everything stays in Python ints — never float).
"""

from __future__ import annotations

import re
from collections import Counter

_BOXED_RE = re.compile(r"\\boxed\s*")
_ANSWER_PHRASE_RE = re.compile(
    r"(?:final answer|answer)\s*(?:is|:|=)\s*\$?(-?[\d,]+)", re.IGNORECASE
)
_INT_RE = re.compile(r"-?\d[\d,]*")
_FRAC_RE = re.compile(r"\\[dt]?frac\s*\{(-?[\d,]+)\}\s*\{(-?[\d,]+)\}")


def _parse_int_like(s: str):
    """Parse a string that should contain one integer; return int or None."""
    if s is None:
        return None
    s = s.strip()
    # strip common LaTeX / formatting noise
    s = s.replace("{,}", ",")  # LaTeX thousands separator: 1{,}234
    s = s.replace("$", "").replace("\\!", "").replace("\\,", "").replace("\\;", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\text\s*\{[^}]*\}", "", s)
    s = re.sub(r"\\mathrm\s*\{[^}]*\}", "", s)
    s = s.strip().rstrip(".。 ")

    # \frac{a}{b} that divides evenly
    m = _FRAC_RE.search(s)
    if m:
        try:
            a = int(m.group(1).replace(",", ""))
            b = int(m.group(2).replace(",", ""))
            if b != 0 and a % b == 0:
                return a // b
        except ValueError:
            pass

    s = s.replace(",", "").replace(" ", "")
    # "42.0" -> 42 ; "42.5" -> None
    m = re.fullmatch(r"(-?\d+)(?:\.0*)?", s)
    if m:
        return int(m.group(1))
    # last resort: first integer inside whatever is left
    m = _INT_RE.search(s)
    if m:
        try:
            return int(m.group(0).replace(",", ""))
        except ValueError:
            return None
    return None


def _last_boxed_content(text: str):
    """Return the content of the last \\boxed{...} using balanced-brace parsing."""
    last = None
    for m in _BOXED_RE.finditer(text):
        i = m.end()
        if i < len(text) and text[i] == "{":
            depth, j = 0, i
            while j < len(text):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if depth == 0:
                last = text[i + 1 : j]
        else:  # \boxed 42  (no braces)
            m2 = _INT_RE.match(text[i:].lstrip())
            if m2:
                last = m2.group(0)
    return last


def extract_final_int(text: str):
    """Extract the final integer answer from a model response. None if absent."""
    if not text:
        return None

    boxed = _last_boxed_content(text)
    if boxed is not None:
        val = _parse_int_like(boxed)
        if val is not None:
            return val

    phrases = _ANSWER_PHRASE_RE.findall(text)
    if phrases:
        val = _parse_int_like(phrases[-1])
        if val is not None:
            return val

    ints = _INT_RE.findall(text)
    if ints:
        try:
            return int(ints[-1].replace(",", ""))
        except ValueError:
            return None
    return None


def extract_boxed_int(text: str):
    """Strict variant: only accept an integer coming from \\boxed{...}.

    Used to filter rejection-sampling data for SFT (higher precision).
    """
    if not text:
        return None
    boxed = _last_boxed_content(text)
    if boxed is None:
        return None
    return _parse_int_like(boxed)


def majority_vote(answers: list):
    """Majority vote over a list of Optional[int]; ties broken by first seen.

    Returns None only when every element is None.
    """
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    counts = Counter(valid)
    best = max(counts.items(), key=lambda kv: (kv[1], -valid.index(kv[0])))
    return best[0]
