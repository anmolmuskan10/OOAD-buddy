"""
validators/gemini_validator.py
──────────────────────────────────────────────────────────────────────────────
Gemini AI Validator for ALL diagram types:
  - Class Diagram
  - Use Case Diagram
  - Sequence Diagram

Model: gemini-2.5-flash-lite
Auto fallback: gemini-2.0-flash-lite → gemini-2.0-flash
Rate limit (429): auto wait + retry
"""

import os
import json
import logging
import re
import time
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional

_log = logging.getLogger(__name__)

_MODELS = [
    "gemini-2.5-flash-lite",   # primary
    "gemini-2.0-flash-lite",   # fallback 1
    "gemini-2.0-flash",        # fallback 2
]

_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
_TIMEOUT_SECONDS = 60
_RETRY_WAIT      = 40


def _get_api_key() -> Optional[str]:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    return key if key else None


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS — alag diagram type ke liye alag prompt
# ─────────────────────────────────────────────────────────────────────────────

def _prompt_class(scenario: str, shapes: List[Dict]) -> str:
    return f"""You are an expert UML Class Diagram validator.

## TASK
Validate the class diagram (JSON shapes) against the scenario. Return ONLY valid JSON.

## SCENARIO
{scenario}

## DIAGRAM SHAPES
{json.dumps(shapes, indent=2)}

## RULES TO CHECK
1. MISSING_CLASS        — Important nouns in scenario must be classes.
2. EXTRA_CLASS          — Classes not in scenario (warning).
3. WRONG_RELATIONSHIP   — Wrong relationship type vs scenario (association/aggregation/composition/generalization).
4. MISSING_RELATIONSHIP — Relationship in scenario not drawn.
5. MISSING_MULTIPLICITY — Associations must have multiplicity on both ends.
6. WRONG_INHERITANCE    — Inheritance arrow direction reversed (child points TO parent).
7. MISSING_ATTRIBUTE    — Class missing key attribute from scenario.
8. MISSING_METHOD       — Class missing important method from scenario.
9. DUPLICATE_CLASS      — Same class name appears twice.
10. CIRCULAR_INHERITANCE — A inherits B and B inherits A.
11. EMPTY_CLASS_NAME    — Class has no name or placeholder like "Class 1".
12. SELF_ASSOCIATION    — Class connected to itself (warn unless scenario says so).

## SEVERITY
ERROR = must fix | WARNING = should fix | INFO = suggestion

## RESPONSE FORMAT (JSON only, no markdown)
{{
  "errors": [
    {{
      "error_type": "RULE_CODE",
      "severity": "ERROR|WARNING|INFO",
      "element": "ClassName",
      "description": "What is wrong",
      "suggestion": "How to fix"
    }}
  ],
  "score": 0-100,
  "summary": "e.g. 2 errors, 1 warning"
}}

If correct: {{"errors": [], "score": 100, "summary": "Diagram is correct"}}
Only report issues you are CONFIDENT about. Do NOT invent errors.
"""


def _prompt_usecase(scenario: str, shapes: List[Dict]) -> str:
    return f"""You are an expert UML Use Case Diagram validator.

## TASK
Validate the use case diagram (JSON shapes) against the scenario. Return ONLY valid JSON.

## SCENARIO
{scenario}

## DIAGRAM SHAPES
{json.dumps(shapes, indent=2)}

## RULES TO CHECK
1. MISSING_ACTOR         — Every person/system in scenario must be an actor.
2. EXTRA_ACTOR           — Actor not mentioned in scenario (warning).
3. MISSING_USE_CASE      — Every action/function in scenario must be a use case.
4. EXTRA_USE_CASE        — Use case not in scenario (info).
5. DISCONNECTED_ACTOR    — Actor has no line to any use case.
6. ISOLATED_USE_CASE     — Use case has no connection to any actor.
7. MISSING_SYSTEM_BOUNDARY — System boundary box is missing.
8. WRONG_RELATIONSHIP    — include/extend/generalization used incorrectly.
9. MISSING_VERB_IN_USE_CASE — Use case name missing action verb (e.g. "Login" not "User Login").
10. DUPLICATE_ACTOR      — Same actor name appears twice.
11. DUPLICATE_USE_CASE   — Same use case name appears twice.
12. ACTOR_NOT_IN_BOUNDARY — Use cases should be inside system boundary.

## SEVERITY
ERROR = must fix | WARNING = should fix | INFO = suggestion

## RESPONSE FORMAT (JSON only, no markdown)
{{
  "errors": [
    {{
      "error_type": "RULE_CODE",
      "severity": "ERROR|WARNING|INFO",
      "element": "ActorOrUseCaseName",
      "description": "What is wrong",
      "suggestion": "How to fix"
    }}
  ],
  "score": 0-100,
  "summary": "e.g. 2 errors, 1 warning"
}}

If correct: {{"errors": [], "score": 100, "summary": "Diagram is correct"}}
Only report issues you are CONFIDENT about. Do NOT invent errors.
"""


def _prompt_sequence(scenario: str, shapes: List[Dict]) -> str:
    return f"""You are an expert UML Sequence Diagram validator.

## TASK
Validate the sequence diagram (JSON shapes) against the scenario. Return ONLY valid JSON.

## SCENARIO
{scenario}

## DIAGRAM SHAPES
{json.dumps(shapes, indent=2)}

## RULES TO CHECK
1. MISSING_LIFELINE      — Every participant/object in scenario must have a lifeline.
2. EXTRA_LIFELINE        — Lifeline not in scenario (warning).
3. MISSING_MESSAGE       — Important interaction in scenario not shown as message arrow.
4. WRONG_MESSAGE_ORDER   — Messages are in wrong chronological order vs scenario.
5. MISSING_RETURN        — A call message has no return/response message.
6. INVALID_MESSAGE_SOURCE — Message arrow starts from non-existent lifeline.
7. INVALID_MESSAGE_TARGET — Message arrow ends at non-existent lifeline.
8. ISOLATED_LIFELINE     — Lifeline sends/receives no messages.
9. EMPTY_LIFELINE_NAME   — Lifeline has no label.
10. MISSING_ACTIVATION   — Lifeline that receives messages has no activation box.
11. SELF_MESSAGE         — Object sends message to itself (warn unless intentional).
12. MISSING_ALT_FRAGMENT — Conditional logic in scenario not shown as alt/opt fragment.

## SEVERITY
ERROR = must fix | WARNING = should fix | INFO = suggestion

## RESPONSE FORMAT (JSON only, no markdown)
{{
  "errors": [
    {{
      "error_type": "RULE_CODE",
      "severity": "ERROR|WARNING|INFO",
      "element": "LifelineOrMessageName",
      "description": "What is wrong",
      "suggestion": "How to fix"
    }}
  ],
  "score": 0-100,
  "summary": "e.g. 2 errors, 1 warning"
}}

If correct: {{"errors": [], "score": 100, "summary": "Diagram is correct"}}
Only report issues you are CONFIDENT about. Do NOT invent errors.
"""


def _build_prompt(diagram_type: str, scenario: str, shapes: List[Dict]) -> str:
    dt = diagram_type.lower()
    if "class" in dt:
        return _prompt_class(scenario, shapes)
    elif "usecase" in dt or "use_case" in dt or "use case" in dt:
        return _prompt_usecase(scenario, shapes)
    elif "sequence" in dt:
        return _prompt_sequence(scenario, shapes)
    else:
        return _prompt_class(scenario, shapes)  # default fallback


# ─────────────────────────────────────────────────────────────────────────────
# HTTP call
# ─────────────────────────────────────────────────────────────────────────────

def _call_model(prompt: str, api_key: str, model: str) -> Optional[Dict]:
    url = f"{_GEMINI_API_BASE}/{model}:generateContent?key={api_key}"
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature":     0.1,
            "maxOutputTokens": 8192,
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        if e.code == 429:
            _log.warning("Model %s: rate limit — waiting %ss then retry...", model, _RETRY_WAIT)
            time.sleep(_RETRY_WAIT)
            try:
                with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
                    body = resp.read().decode("utf-8")
            except Exception as e2:
                _log.warning("Model %s: retry failed: %s", model, e2)
                return None
        elif e.code == 404:
            _log.warning("Model %s: not found, trying next...", model)
            return None
        else:
            _log.error("Model %s: HTTP %s: %s", model, e.code, err_body[:500])
            return None
    except Exception as e:
        _log.warning("Model %s: failed: %s", model, e)
        return None

    try:
        outer = json.loads(body)
        text  = outer["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
        _log.warning("Model %s: parse error: %s", model, e)
        return None

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$",        "", text.rstrip())

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        _log.warning("Model %s: JSON error: %s", model, e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def validate_with_gemini(
    scenario:     str,
    shapes:       List[Dict[str, Any]],
    diagram_type: str = "class",
) -> Optional[Dict[str, Any]]:
    """
    Validate any diagram type using Gemini AI.
    diagram_type: 'class' | 'usecase' | 'sequence'
    Returns structured validation result or None.
    """
    api_key = _get_api_key()
    if not api_key:
        _log.warning("GEMINI_API_KEY not set — skipping AI validation")
        return None

    prompt = _build_prompt(diagram_type, scenario, shapes)

    for model in _MODELS:
        _log.info("Trying Gemini model: %s (diagram: %s)", model, diagram_type)
        result = _call_model(prompt, api_key, model)
        if result:
            _log.info("Gemini model %s succeeded!", model)
            raw_errors = result.get("errors", [])
            score      = int(result.get("score", 50))
            summary    = result.get("summary", "Gemini validation complete")

            errors, warnings, info = [], [], []
            for e in raw_errors:
                sev  = str(e.get("severity", "ERROR")).upper()
                item = {
                    "error_type":  str(e.get("error_type",  "UNKNOWN")),
                    "severity":    sev,
                    "element":     str(e.get("element",     "")),
                    "description": str(e.get("description", "")),
                    "suggestion":  str(e.get("suggestion",  "")),
                }
                if sev == "WARNING":   warnings.append(item)
                elif sev == "INFO":    info.append(item)
                else:                  errors.append(item)

            return {
                "is_valid":     len(errors) == 0,
                "score":        score,
                "summary":      summary,
                "errors":       errors,
                "warnings":     warnings,
                "info":         info,
                "total_issues": len(raw_errors),
                "source":       "gemini",
            }

    _log.error("All Gemini models failed!")
    return None

