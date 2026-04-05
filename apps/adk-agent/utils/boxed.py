# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Boxed content extraction utilities.

Extracts content from LaTeX-style \\boxed{} expressions, handling nested braces,
escaped characters, and incomplete expressions.
"""

import re
from typing import Optional

# Values that should be treated as invalid/empty answers
BLACKLISTED_VALUES = {"?", "??", "???", "\uff1f", "\u2026\u2026", "\u2026", "...", "unknown"}

_BOXED_RE = re.compile(r"\\boxed\b", re.DOTALL)


def extract_boxed_content(text: str) -> str:
    r"""
    Extract the content of the last \boxed{...} occurrence in the given text.

    Supports:
      - Arbitrary levels of nested braces
      - Escaped braces (\{ and \})
      - Whitespace between \boxed and the opening brace
      - Empty content inside braces
      - Incomplete boxed expressions (extracts to end of string as fallback)

    Args:
        text: Input text that may contain \boxed{...} expressions

    Returns:
        The extracted boxed content, or empty string if no match is found.
    """
    if not text:
        return ""

    last_result: Optional[str] = None
    i = 0
    n = len(text)

    while True:
        m = _BOXED_RE.search(text, i)
        if not m:
            break
        j = m.end()

        # Skip any whitespace after \boxed
        while j < n and text[j].isspace():
            j += 1

        # Require that the next character is '{'
        if j >= n or text[j] != "{":
            i = j
            continue

        # Parse the brace content manually to handle nesting and escapes
        depth = 0
        k = j
        escaped = False
        found_closing = False
        while k < n:
            ch = text[k]
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    last_result = text[j + 1 : k]
                    i = k + 1
                    found_closing = True
                    break
            k += 1

        # If we didn't find a closing brace, this is an incomplete boxed
        # Store it as the last result (will be overwritten if we find more)
        if not found_closing and depth > 0:
            last_result = text[j + 1 : n]
            i = k
        elif not found_closing:
            i = j + 1

    if last_result is None or last_result.strip() in BLACKLISTED_VALUES:
        return ""
    return last_result.strip()
