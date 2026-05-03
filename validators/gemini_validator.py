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

FIX: Shape sanitizer added — strips ToolType. prefix, filters shapes by
     diagram type so Gemini never sees actor/systemBoundary shapes when
     validating a class diagram (and vice versa).
     Prompt headers now explicitly forbid cross-diagram rules.
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
# SHAPE SANITIZER — strips Flutter ToolType prefix, filters by diagram type
# so Gemini cannot be confused by cross-diagram shape types
# ─────────────────────────────────────────────────────────────────────────────

# Flutter ToolType enum value → clean readable name for Gemini
_TOOLTYPE_MAP = {
    "classfullshape":       "class",
    "classshape":           "class",
    "generalization":       "generalization_arrow",
    "association":          "association_arrow",
    "aggregation":          "aggregation_arrow",
    "composition":          "composition_arrow",
    "dependency":           "dependency_arrow",
    "dashedarrow":          "dashed_arrow",
    "dashedopenarow":       "dashed_open_arrow",
    "dottedarrow":          "dotted_arrow",
    "excludearrow":         "include_extend_arrow",
    "arrow":                "arrow",
    "straightline":         "line",
    "actor":                "actor",
    "usecase":              "use_case_oval",
    "systemboundary":       "system_boundary",
    "lifeline":             "lifeline",
    "object":               "object_lifeline",
    "activation":           "activation_box",
    "deletion":             "deletion_marker",
    "fragment":             "combined_fragment",
    "selfmessagearrow":     "self_message_arrow",
    "selfmessagedottedarrow": "self_message_dotted_arrow",
    "text":                 "text_label",
    "circle":               "circle",
}

# Shape types that are valid (expected) in each diagram type
_CLASS_VALID_TYPES = {
    "class", "generalization_arrow", "association_arrow",
    "aggregation_arrow", "composition_arrow", "dependency_arrow",
    "dashed_arrow", "dashed_open_arrow", "dotted_arrow",
    "arrow", "line", "text_label",
}
_USECASE_VALID_TYPES = {
    "actor", "use_case_oval", "system_boundary",
    "include_extend_arrow", "generalization_arrow",
    "dashed_arrow", "dashed_open_arrow", "dotted_arrow",
    "arrow", "line", "text_label",
}
_SEQUENCE_VALID_TYPES = {
    "lifeline", "object_lifeline", "actor", "activation_box",
    "deletion_marker", "combined_fragment",
    "arrow", "dashed_arrow", "dotted_arrow",
    "self_message_arrow", "self_message_dotted_arrow",
    "line", "text_label",
}

_VALID_TYPES_BY_DIAGRAM = {
    "class":    _CLASS_VALID_TYPES,
    "usecase":  _USECASE_VALID_TYPES,
    "sequence": _SEQUENCE_VALID_TYPES,
}


def _clean_type(raw_type: str) -> str:
    """Strip 'ToolType.' prefix and return a readable name."""
    t = str(raw_type).strip().lower()
    if "." in t:
        t = t.split(".")[-1]            # 'ToolType.classFullShape' → 'classfullshape'
    t = re.sub(r"[_\-]", "", t)        # drop separators before lookup
    return _TOOLTYPE_MAP.get(t, t)


def _sanitize_shapes(shapes: List[Dict], diagram_type: str) -> List[Dict]:
    """
    Prepare shapes for Gemini:
    1. Clean 'type' field — strip ToolType. prefix, map to readable name.
    2. Remove shapes that don't belong to this diagram type (they confuse Gemini).
    3. Keep only fields Gemini needs; drop internal Flutter state fields.
    """
    valid_types = _VALID_TYPES_BY_DIAGRAM.get(diagram_type)
    cleaned = []

    for s in shapes:
        clean_t = _clean_type(str(s.get("type", "")))

        # Filter out foreign-diagram shapes UNLESS they carry connection info
        if valid_types and clean_t not in valid_types:
            has_conn = any(s.get(f) for f in ("from", "to", "startLifeline", "endLifeline"))
            if not has_conn:
                continue  # skip this shape — it doesn't belong here

        # Build a minimal, clean dict for Gemini
        entry: Dict[str, Any] = {"type": clean_t}
        for field in ("text", "label", "name", "id", "from", "to",
                      "startLifeline", "endLifeline", "lifelineRef"):
            val = s.get(field)
            if val is not None and str(val).strip().lower() not in ("", "none", "null", "undefined"):
                entry[field] = val
        cleaned.append(entry)

    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS — alag diagram type ke liye alag prompt
# ─────────────────────────────────────────────────────────────────────────────

def _prompt_class(scenario: str, shapes: List[Dict]) -> str:
    return f"""You are an expert UML Class Diagram validator.

## TASK
This is a CLASS DIAGRAM. Validate it using ONLY class diagram rules. Return ONLY valid JSON.

## ABSOLUTE RESTRICTIONS — violating these makes your response wrong:
- Do NOT check for actors, stick figures, system boundaries, use cases, lifelines, or messages.
- Do NOT report MISSING_ACTOR, MISSING_SYSTEM_BOUNDARY, MISSING_USE_CASE, MISSING_LIFELINE.
- Do NOT apply use case or sequence diagram rules of any kind.
- ONLY apply the 12 class diagram rules listed below.

## SCENARIO
{scenario}

## DIAGRAM SHAPES
{json.dumps(shapes, indent=2)}

## RULES TO CHECK (class diagram ONLY)
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
This is a USE CASE DIAGRAM. Validate it using ONLY use case diagram rules. Return ONLY valid JSON.

## ABSOLUTE RESTRICTIONS:
- Do NOT check for class boxes, attributes, methods, lifelines, or activation bars.
- Do NOT report MISSING_CLASS, MISSING_ATTRIBUTE, MISSING_METHOD, MISSING_LIFELINE.
- Do NOT apply class diagram or sequence diagram rules of any kind.

## SCENARIO
{scenario}

## DIAGRAM SHAPES
{json.dumps(shapes, indent=2)}

## RULES TO CHECK (use case diagram ONLY)
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
This is a SEQUENCE DIAGRAM. Validate it using ONLY sequence diagram rules. Return ONLY valid JSON.

## ABSOLUTE RESTRICTIONS:
- Do NOT check for class boxes, actors (unless they are lifelines), system boundaries, or use cases.
- Do NOT report MISSING_CLASS, MISSING_ACTOR, MISSING_SYSTEM_BOUNDARY, MISSING_USE_CASE.
- Do NOT apply class diagram or use case diagram rules of any kind.

## SCENARIO
{scenario}

## DIAGRAM SHAPES
{json.dumps(shapes, indent=2)}

## RULES TO CHECK (sequence diagram ONLY)
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

    # ── Sanitize shapes BEFORE building prompt ────────────────────────────────
    # This strips ToolType. prefix and removes cross-diagram shapes so Gemini
    # doesn't get confused (e.g. actor shapes confusing a class diagram check).
    clean_shapes = _sanitize_shapes(shapes, diagram_type)
    _log.info("Sanitized shapes: %d → %d (diagram: %s)", len(shapes), len(clean_shapes), diagram_type)

    prompt = _build_prompt(diagram_type, scenario, clean_shapes)

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


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE-AWARE PROMPT — jab shapes empty ho aur image available ho
# ─────────────────────────────────────────────────────────────────────────────

def _build_image_prompt(diagram_type: str, scenario: str) -> str:
    """
    Jab shapes empty ho aur image available ho — image ko directly analyze karo.
    Shapes ke baghair Gemini ko image dekhni chahiye taake accurately validate kare.
    """
    dt = diagram_type.lower()

    if "usecase" in dt or "use_case" in dt:
        rules = """1. MISSING_ACTOR — Every person/system in scenario must be an actor (stick figure).
2. EXTRA_ACTOR — Actor not mentioned in scenario (warning).
3. MISSING_USE_CASE — Every action/function in scenario must be a use case oval.
4. EXTRA_USE_CASE — Use case oval not in scenario (info).
5. DISCONNECTED_ACTOR — Actor has no line to any use case.
6. ISOLATED_USE_CASE — Use case has no connection to any actor.
7. MISSING_SYSTEM_BOUNDARY — Only report if you CANNOT SEE a rectangle/box enclosing use cases. If rectangle is visible → do NOT report.
8. WRONG_RELATIONSHIP — include/extend/generalization used incorrectly.
9. MISSING_VERB_IN_USE_CASE — Use case name missing action verb.
10. DUPLICATE_ACTOR — Same actor name appears twice.
11. DUPLICATE_USE_CASE — Same use case name appears twice."""
        dtype_label = "USE CASE"

    elif "sequence" in dt:
        rules = """1. MISSING_LIFELINE — Every participant in scenario must have a lifeline.
2. MISSING_MESSAGE — Important interaction in scenario not shown as message arrow.
3. WRONG_MESSAGE_ORDER — Messages in wrong chronological order.
4. MISSING_RETURN — A call message has no return message.
5. ISOLATED_LIFELINE — Lifeline sends/receives no messages.
6. EMPTY_LIFELINE_NAME — Lifeline has no label."""
        dtype_label = "SEQUENCE"

    else:
        rules = """1. MISSING_CLASS — Important nouns in scenario must be class boxes.
2. EXTRA_CLASS — Class not in scenario (warning).
3. WRONG_RELATIONSHIP — Wrong arrow type used.
4. MISSING_RELATIONSHIP — Relationship in scenario not drawn.
5. MISSING_MULTIPLICITY — Associations missing multiplicity labels.
6. EMPTY_CLASS_NAME — Class has no name."""
        dtype_label = "CLASS"

    return f"""You are an expert UML {dtype_label} Diagram validator. You are given an IMAGE of the diagram.

## CRITICAL INSTRUCTION
Look carefully at the ACTUAL IMAGE provided. Validate ONLY what you can SEE.
- If you see a rectangle/box enclosing the use cases → system boundary EXISTS, do NOT report MISSING_SYSTEM_BOUNDARY.
- If you see stick figures → actors EXIST, do NOT report them missing.
- ONLY report something as MISSING if it is genuinely absent from the image.

## SCENARIO
{scenario}

## RULES TO CHECK ({dtype_label} diagram ONLY)
{rules}

## SEVERITY
ERROR = must fix | WARNING = should fix | INFO = suggestion

## RESPONSE FORMAT (JSON only, no markdown)
{{
  "errors": [
    {{
      "error_type": "RULE_CODE",
      "severity": "ERROR|WARNING|INFO",
      "element": "element name",
      "description": "What is wrong",
      "suggestion": "How to fix"
    }}
  ],
  "score": 0-100,
  "summary": "brief summary"
}}

If correct: {{"errors": [], "score": 100, "summary": "Diagram is correct"}}
Only report issues you are CONFIDENT about from what you SEE. Do NOT invent errors."""


def _call_model_with_image(prompt: str, image_b64: str, mime_type: str, api_key: str, model: str) -> Optional[Dict]:
    """Gemini Vision — image + text prompt dono bhejo."""
    url = f"{_GEMINI_API_BASE}/{model}:generateContent?key={api_key}"
    payload = json.dumps({
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime_type, "data": image_b64}},
            ]
        }],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 8192},
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
            _log.warning("Vision model %s: rate limit — waiting %ss", model, _RETRY_WAIT)
            time.sleep(_RETRY_WAIT)
            try:
                with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
                    body = resp.read().decode("utf-8")
            except Exception as e2:
                _log.warning("Vision retry failed: %s", e2)
                return None
        else:
            _log.error("Vision model %s: HTTP %s: %s", model, e.code, err_body[:300])
            return None
    except Exception as e:
        _log.warning("Vision model %s failed: %s", model, e)
        return None

    try:
        outer = json.loads(body)
        text  = outer["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
        _log.warning("Vision parse error: %s", e)
        return None

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.rstrip())

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        _log.warning("Vision JSON error: %s | text: %s", e, text[:200])
        return None


def validate_with_gemini_image(
    scenario:     str,
    image_b64:    str,
    mime_type:    str = "image/png",
    diagram_type: str = "class",
) -> Optional[Dict[str, Any]]:
    """
    Image-based validation — shapes ki jagah actual diagram image bhejo Gemini ko.
    Yeh tab use hota hai jab user gallery se image upload kare (canvas ke baghair).

    diagram_type: 'class' | 'usecase' | 'sequence'
    Returns structured validation result or None.
    """
    api_key = _get_api_key()
    if not api_key:
        _log.warning("GEMINI_API_KEY not set — skipping image validation")
        return None

    # Vision-capable models only (gemini-2.5-flash-lite supports vision)
    vision_models = ["gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-2.0-flash-lite"]

    prompt = _build_image_prompt(diagram_type, scenario)

    for model in vision_models:
        _log.info("Trying Vision model: %s (diagram: %s)", model, diagram_type)
        result = _call_model_with_image(prompt, image_b64, mime_type, api_key, model)
        if result:
            _log.info("Vision model %s succeeded!", model)
            raw_errors = result.get("errors", [])
            score      = int(result.get("score", 50))
            summary    = result.get("summary", "Image validation complete")

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
                if sev == "WARNING":  warnings.append(item)
                elif sev == "INFO":   info.append(item)
                else:                 errors.append(item)

            return {
                "is_valid":     len(errors) == 0,
                "score":        score,
                "summary":      summary,
                "errors":       errors,
                "warnings":     warnings,
                "info":         info,
                "total_issues": len(raw_errors),
                "source":       "gemini-vision",
                "validation_mode": "gemini",
            }

    _log.error("All Vision models failed!")
    return None
