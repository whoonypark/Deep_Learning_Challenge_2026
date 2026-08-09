"""Prompt construction for the math challenge.

Single source of truth for prompts so training and inference always match.
"""

SYSTEM_PROMPT = (
    "You are a careful competition mathematician. Solve the given problem by "
    "reasoning step by step, then finish with your final answer in the form "
    "\\boxed{N} where N is a single integer."
)

USER_SUFFIX = (
    "\n\nSolve the problem step by step. The final answer is always a single "
    "integer. End your response with \\boxed{<integer>}."
)


def build_messages(question: str) -> list:
    """Return an OpenAI/Qwen-style chat message list for one question."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip() + USER_SUFFIX},
    ]
