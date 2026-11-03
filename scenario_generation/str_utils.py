from __future__ import annotations

import json
import re
from typing import Any

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    # ```json ... ``` or ``` ... ```
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_first_json_array(s: str) -> str | None:
    # Try direct first
    s1 = _strip_code_fences(s)
    try:
        obj = json.loads(s1)
        if isinstance(obj, list):
            return s1
    except Exception:
        pass
    # Fallback: find first top-level array
    m = re.search(r"\[\s*{[\s\S]*}\s*\]", s1)
    if m:
        return m.group(0)
    return None


def _extract_first_json_dict(s: str) -> str | None:
    s1 = _strip_code_fences(s)

    # Try direct parse first
    try:
        obj = json.loads(s1)
        if isinstance(obj, dict):
            return s1
    except Exception:
        pass

    # Fallback: find first top-level dict
    m = re.search(r"{[\s\S]*}", s1)
    if m:
        candidate = m.group(0)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return candidate
        except Exception:
            pass

    return None

def _type_matches(expected: str, value: Any) -> bool:
    e = expected.strip().lower()

    # Common aliases
    if e in ("string", "str"):
        return isinstance(value, str)
    if e in ("int", "integer"):
        return isinstance(value, int) and not isinstance(value, bool)
    if e in ("float",):
        return isinstance(value, float)
    if e in ("number",):
        return (isinstance(value, (int, float)) and not isinstance(value, bool))
    if e in ("bool", "boolean"):
        return isinstance(value, bool)
    if e in ("array", "list"):
        return isinstance(value, list)
    if e in ("object", "dict"):
        return isinstance(value, dict)

    # Patterns like "list of strings", "list of numbers", "list[str]"
    m = re.match(r"(list|array)\s+of\s+(\w+)", e)
    if m and isinstance(value, list):
        subtype = m.group(2)
        if subtype in ("strings", "string", "str"):
            return all(isinstance(x, str) for x in value)
        if subtype in ("numbers", "number"):
            return all((isinstance(x, (int, float)) and not isinstance(x, bool)) for x in value)
        if subtype in ("ints", "integers", "int", "integer"):
            return all((isinstance(x, int) and not isinstance(x, bool)) for x in value)
        if subtype in ("floats", "float"):
            return all(isinstance(x, float) for x in value)
        if subtype in ("bools", "booleans", "bool", "boolean"):
            return all(isinstance(x, bool) for x in value)
        if subtype in ("objects", "dicts", "object", "dict"):
            return all(isinstance(x, dict) for x in value)
        return True  # unknown subtype; can't strictly verify

    # Patterns like "list[str]" or "list[int]"
    m = re.match(r"(list|array)\[(\w+)\]", e)
    if m and isinstance(value, list):
        subtype = m.group(2)
        return _type_matches(f"list of {subtype}s", value)

    # Unknown spec; accept
    return True
