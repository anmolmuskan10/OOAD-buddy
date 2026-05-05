"""
validators/openai_validator.py
──────────────────────────────────────────────────────────────────────────────
OpenAI AI Validator for ALL diagram types:
  - Class Diagram
  - Use Case Diagram
  - Sequence Diagram

Model: gpt-4o
Rate limit (429): auto wait + retry

FIX: Shape sanitizer added — strips ToolType. prefix, filters shapes by
     diagram type so OpenAI never sees actor/systemBoundary shapes when
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
    "gpt-4o",
]

_OPENAI_API_BASE = "https://api.openai.com/v1"
_TIMEOUT_SECONDS = 60
_RETRY_WAIT      = 40


def _get_api_key() -> Optional[str]:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    return key if key else None


# ─────────────────────────────────────────────────────────────────────────────
# SHAPE SANITIZER — strips Flutter ToolType prefix, filters by diagram type
# so OpenAI cannot be confused by cross-diagram shape types
# ─────────────────────────────────────────────────────────────────────────────

# Flutter ToolType enum value → clean readable name for OpenAI
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
    Prepare shapes for OpenAI:
    1. Clean 'type' field — strip ToolType. prefix, map to readable name.
    2. Remove shapes that don't belong to this diagram type (they confuse OpenAI).
    3. Keep only fields OpenAI needs; drop internal Flutter state fields.
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

        # Build a minimal, clean dict for OpenAI
        entry: Dict[str, Any] = {"type": clean_t}
        for field in ("text", "label", "name", "id", "from", "to",
                      "startLifeline", "endLifeline", "lifelineRef"):
            val = s.get(field)
            if val is not None and str(val).strip().lower() not in ("", "none", "null", "undefined"):
                # classFullShape text = "ClassName\n---attrs---\n..." — send only class name
                if field == "text" and "\n" in str(val):
                    val = str(val).split("\n")[0].strip()
                entry[field] = val
        cleaned.append(entry)

    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK AUTO-FIX BUILDER
# Jab OpenAI fixable: true nahi deta, hum error_type se apna fix banate hain
# ─────────────────────────────────────────────────────────────────────────────

def _build_fallback_fix(error_type: str, element: str, raw_error: dict, diagram_type: str) -> dict:
    """
    OpenAI ka auto_fix agar incomplete/missing ho — error_type se fallback fix banao.
    Ye ensure karta hai ke common fixable errors hamesha fixable rahein.
    """
    et = error_type.upper()
    desc = str(raw_error.get("description", "")).lower()
    suggestion = str(raw_error.get("suggestion", ""))

    # ── CLASS DIAGRAM ─────────────────────────────────────────────────────────
    if "MISSING_CLASS" in et:
        return {"fixable": True, "action": "add_shape", "shape_type": "class", "name": element or "NewClass"}

    if "DUPLICATE_CLASS" in et:
        return {"fixable": True, "action": "merge_shapes", "name": element}

    if "EMPTY_CLASS_NAME" in et:
        return {"fixable": True, "action": "rename_shape", "name": element or "ClassName"}

    if "MISSING_RELATIONSHIP" in et or "MISSING_RELATIONSHIP" in et:
        # Try to parse from_element and to_element from description/suggestion
        from_el, to_el = _parse_from_to(desc + " " + suggestion.lower())
        arrow = _guess_arrow_type(desc + " " + suggestion.lower(), diagram_type)
        if from_el and to_el:
            return {"fixable": True, "action": "add_arrow",
                    "from_element": from_el, "to_element": to_el, "arrow_type": arrow}
        # element might be "ClassA — ClassB"
        if "—" in element or "-" in element:
            parts = element.replace("—", "-").split("-")
            if len(parts) >= 2:
                return {"fixable": True, "action": "add_arrow",
                        "from_element": parts[0].strip(), "to_element": parts[1].strip(),
                        "arrow_type": arrow}
        return {"fixable": False}

    if "WRONG_RELATIONSHIP" in et:
        from_el, to_el = _parse_from_to(desc + " " + suggestion.lower())
        arrow = _guess_arrow_type(desc + " " + suggestion.lower(), diagram_type)
        if from_el and to_el:
            return {"fixable": True, "action": "change_arrow_type",
                    "from_element": from_el, "to_element": to_el, "arrow_type": arrow}
        return {"fixable": False}

    if "MISSING_MULTIPLICITY" in et:
        from_el, to_el = _parse_from_to(desc + " " + suggestion.lower())
        if from_el and to_el:
            return {"fixable": True, "action": "add_label",
                    "from_element": from_el, "to_element": to_el,
                    "multiplicity_from": "1", "multiplicity_to": "*"}
        return {"fixable": False}

    # ── USE CASE DIAGRAM ──────────────────────────────────────────────────────
    if "MISSING_ACTOR" in et:
        return {"fixable": True, "action": "add_shape", "shape_type": "actor", "name": element or "Actor"}

    if "MISSING_USE_CASE" in et:
        return {"fixable": True, "action": "add_shape", "shape_type": "use_case_oval", "name": element or "UseCase"}

    if "MISSING_SYSTEM_BOUNDARY" in et:
        return {"fixable": True, "action": "add_boundary", "shape_type": "system_boundary"}

    if "DISCONNECTED_ACTOR" in et:
        return {"fixable": True, "action": "add_arrow",
                "from_element": element, "to_element": "", "arrow_type": "association"}

    if "ISOLATED_USE_CASE" in et:
        return {"fixable": True, "action": "add_arrow",
                "from_element": "", "to_element": element, "arrow_type": "association"}

    if "DUPLICATE_ACTOR" in et or "DUPLICATE_USE_CASE" in et:
        return {"fixable": True, "action": "merge_shapes", "name": element}

    if "MISSING_VERB_IN_USE_CASE" in et:
        # Try to add a verb from suggestion
        name = element or "DoAction"
        if suggestion:
            import re as _re
            m = _re.search(r"'([^']+)'", suggestion)
            if m: name = m.group(1)
        return {"fixable": True, "action": "rename_shape", "name": name}

    # ── SEQUENCE DIAGRAM ──────────────────────────────────────────────────────
    if "MISSING_LIFELINE" in et:
        return {"fixable": True, "action": "add_shape", "shape_type": "lifeline", "name": element or "Participant"}

    if "EMPTY_LIFELINE_NAME" in et:
        return {"fixable": True, "action": "rename_shape", "name": element or "Participant"}

    if "MISSING_MESSAGE" in et or "MISSING_RETURN" in et:
        from_el, to_el = _parse_from_to(desc + " " + suggestion.lower())
        arrow = "dashed_arrow" if "return" in et.lower() or "response" in desc else "arrow"
        if from_el and to_el:
            return {"fixable": True, "action": "add_arrow",
                    "from_element": from_el, "to_element": to_el,
                    "message_label": element or "", "arrow_type": arrow}
        return {"fixable": False}

    return {"fixable": False}


def _parse_from_to(text: str):
    """Try to extract 'from X to Y' or 'X and Y' or quoted names from text."""
    import re as _re
    # Pattern: from 'X' to 'Y'
    m = _re.search(r"from ['\"]?([A-Za-z][A-Za-z0-9_\s]*?)['\"]? to ['\"]?([A-Za-z][A-Za-z0-9_\s]*?)['\"]?(?:\s|$|\.)", text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # Pattern: between 'X' and 'Y'
    m = _re.search(r"between ['\"]?([A-Za-z][A-Za-z0-9_\s]*?)['\"]? and ['\"]?([A-Za-z][A-Za-z0-9_\s]*?)['\"]?(?:\s|$|\.)", text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # Pattern: 'X' class and the 'Y' class
    m = _re.search(r"['\"]([A-Za-z][A-Za-z0-9_\s]*?)['\"] class and the ['\"]([A-Za-z][A-Za-z0-9_\s]*?)['\"]", text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None


def _guess_arrow_type(text: str, diagram_type: str) -> str:
    """Guess the best arrow type from description/suggestion text."""
    if "composition" in text:   return "composition"
    if "aggregation" in text:   return "aggregation"
    if "generalization" in text or "inherit" in text: return "generalization"
    if "dependency" in text or "depend" in text:      return "dependency"
    if "include" in text:       return "include"
    if "extend" in text:        return "extend"
    return "association"


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

## AUTO-FIX INSTRUCTIONS
For each error, also provide an "auto_fix" object describing exactly how to fix it programmatically.
auto_fix fields:
- "action": one of "add_shape", "delete_shape", "rename_shape", "add_arrow", "delete_arrow", "change_arrow_type", "add_label", "merge_shapes"
- "shape_type": (for add_shape) one of "class"
- "name": new name or label to set
- "from_element": source element name (for arrows)
- "to_element": target element name (for arrows)
- "arrow_type": (for arrows) one of "association", "aggregation", "composition", "generalization", "dependency"
- "multiplicity_from": multiplicity at source end e.g. "1"
- "multiplicity_to": multiplicity at target end e.g. "*"
- "fixable": true if this can be auto-fixed, false if user must fix manually

AUTO-FIX RULES:
- MISSING_CLASS → fixable: true, action: add_shape, shape_type: class, name: <missing class name>
- DUPLICATE_CLASS → fixable: true, action: merge_shapes, name: <class name to keep>
- EMPTY_CLASS_NAME → fixable: true, action: rename_shape, name: <correct name from scenario>
- WRONG_RELATIONSHIP → fixable: true, action: change_arrow_type, from_element, to_element, arrow_type: <correct type>
- MISSING_RELATIONSHIP → fixable: true, action: add_arrow, from_element, to_element, arrow_type: <correct type>
- MISSING_MULTIPLICITY → fixable: true, action: add_label, from_element, to_element, multiplicity_from, multiplicity_to
- WRONG_INHERITANCE → fixable: false (user must fix — reversing arrows changes hierarchy)
- CIRCULAR_INHERITANCE → fixable: false (user must fix — complex structural change)
- EXTRA_CLASS → fixable: false (user may have added intentionally)
- MISSING_ATTRIBUTE / MISSING_METHOD → fixable: false (requires user judgment)
- SELF_ASSOCIATION → fixable: false (user must review)

## RESPONSE FORMAT (JSON only, no markdown)
{{
  "errors": [
    {{
      "error_type": "RULE_CODE",
      "severity": "ERROR|WARNING|INFO",
      "element": "ClassName",
      "description": "What is wrong",
      "suggestion": "How to fix",
      "auto_fix": {{
        "fixable": true,
        "action": "add_shape",
        "shape_type": "class",
        "name": "MissingClassName"
      }}
    }}
  ],
  "score": 0-100,
  "summary": "e.g. 2 errors, 1 warning"
}}

If correct: {{"errors": [], "score": 100, "summary": "Diagram is correct"}}

## STRICT VALIDATION RULES — MUST FOLLOW:
- Only report issues you are CONFIDENT about. Do NOT invent errors.
- STRICT: If an attribute or method is NOT explicitly mentioned in the scenario, do NOT report MISSING_ATTRIBUTE or MISSING_METHOD.
- STRICT: If a relationship is NOT explicitly mentioned in the scenario, do NOT report MISSING_RELATIONSHIP.
- STRICT: If a class is NOT explicitly mentioned in the scenario, do NOT report MISSING_CLASS.
- STRICT: Do NOT assume standard/common attributes (like id, name, date) are required unless scenario says so.
- STRICT: When in doubt about any error, SKIP it — do not report it.
- STRICT: Check EVERY association arrow for multiplicity on BOTH ends. Report MISSING_MULTIPLICITY if any end is missing a label.

## CLASS NAME vs ATTRIBUTE/METHOD — CRITICAL RULES:
- A UML class box has 3 sections: TOP = class name, MIDDLE = attributes, BOTTOM = methods.
- STRICT: ONLY the TOP section is the class name. Do NOT treat attribute names or method names as class names.
- STRICT: Names like 'reviewId', 'staffId', 'patientId', 'staffID' are ATTRIBUTES (middle section) — NOT class names. Never report them as MISSING_CLASS.
- STRICT: Names ending with () like 'submitreview()', 'login()', 'logout()' are METHODS — NOT classes. Never report them as MISSING_CLASS.
- STRICT: Do NOT report SPELLING_MISTAKE if the "corrected" name is actually an attribute or method name visible inside the class box. Example: if class is named 'Review' and 'reviewId' is its attribute, do NOT say 'Review' should be renamed to 'Reviewid' — this is WRONG.
- STRICT: Do NOT confuse camelCase attribute names (reviewId, staffId) with class names even if they look similar to a class name.
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

## AUTO-FIX INSTRUCTIONS
For each error, also provide an "auto_fix" object describing exactly how to fix it programmatically.
auto_fix fields:
- "action": one of "add_shape", "delete_shape", "rename_shape", "add_arrow", "add_boundary", "merge_shapes"
- "shape_type": (for add_shape) one of "actor", "use_case_oval", "system_boundary"
- "name": new name or label to set
- "from_element": source element name (for arrows)
- "to_element": target element name (for arrows)
- "arrow_type": one of "association", "include", "extend", "generalization"
- "fixable": true if this can be auto-fixed, false if user must fix manually

AUTO-FIX RULES:
- MISSING_ACTOR → fixable: true, action: add_shape, shape_type: actor, name: <actor name>
- MISSING_USE_CASE → fixable: true, action: add_shape, shape_type: use_case_oval, name: <use case name>
- MISSING_SYSTEM_BOUNDARY → fixable: true, action: add_boundary, shape_type: system_boundary
- DISCONNECTED_ACTOR → fixable: true, action: add_arrow, from_element: <actor name>, to_element: <use case name>, arrow_type: association
- ISOLATED_USE_CASE → fixable: true, action: add_arrow, from_element: <actor name>, to_element: <use case name>, arrow_type: association
- MISSING_VERB_IN_USE_CASE → fixable: true, action: rename_shape, name: <corrected name with verb>
- DUPLICATE_ACTOR → fixable: true, action: merge_shapes, name: <actor name to keep>
- DUPLICATE_USE_CASE → fixable: true, action: merge_shapes, name: <use case name to keep>
- EXTRA_ACTOR / EXTRA_USE_CASE → fixable: false (user may have added intentionally)
- WRONG_RELATIONSHIP → fixable: false (user must review relationship semantics)
- ACTOR_NOT_IN_BOUNDARY → fixable: false (requires layout restructuring)

## RESPONSE FORMAT (JSON only, no markdown)
{{
  "errors": [
    {{
      "error_type": "RULE_CODE",
      "severity": "ERROR|WARNING|INFO",
      "element": "ActorOrUseCaseName",
      "description": "What is wrong",
      "suggestion": "How to fix",
      "auto_fix": {{
        "fixable": true,
        "action": "add_shape",
        "shape_type": "actor",
        "name": "MissingActorName"
      }}
    }}
  ],
  "score": 0-100,
  "summary": "e.g. 2 errors, 1 warning"
}}

If correct: {{"errors": [], "score": 100, "summary": "Diagram is correct"}}

## STRICT VALIDATION RULES — MUST FOLLOW:
- Only report issues you are CONFIDENT about. Do NOT invent errors.
- STRICT: If an actor is NOT explicitly mentioned in the scenario, do NOT report MISSING_ACTOR.
- STRICT: If a use case is NOT explicitly mentioned in the scenario, do NOT report MISSING_USE_CASE.
- STRICT: If a relationship is NOT explicitly mentioned in the scenario, do NOT report WRONG_RELATIONSHIP.
- STRICT: Do NOT assume actors or use cases that are implied but not written in scenario.
- STRICT: When in doubt about any error, SKIP it — do not report it.
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

## AUTO-FIX INSTRUCTIONS
For each error, also provide an "auto_fix" object describing exactly how to fix it programmatically.
auto_fix fields:
- "action": one of "add_shape", "rename_shape", "add_arrow", "merge_shapes"
- "shape_type": (for add_shape) one of "lifeline", "activation_box", "combined_fragment"
- "name": new name or label to set
- "from_element": source lifeline name (for arrows)
- "to_element": target lifeline name (for arrows)
- "message_label": label to put on the arrow
- "arrow_type": one of "arrow" (solid call), "dashed_arrow" (return/response)
- "fixable": true if this can be auto-fixed, false if user must fix manually

AUTO-FIX RULES:
- MISSING_LIFELINE → fixable: true, action: add_shape, shape_type: lifeline, name: <participant name>
- EMPTY_LIFELINE_NAME → fixable: true, action: rename_shape, name: <correct name from scenario>
- MISSING_MESSAGE → fixable: true, action: add_arrow, from_element, to_element, message_label, arrow_type: arrow
- MISSING_RETURN → fixable: true, action: add_arrow, from_element: <receiver>, to_element: <sender>, message_label: <return label>, arrow_type: dashed_arrow
- WRONG_MESSAGE_ORDER → fixable: false (reordering requires full diagram restructure)
- ISOLATED_LIFELINE → fixable: false (user must decide which messages to add)
- EXTRA_LIFELINE → fixable: false (user may have added intentionally)
- INVALID_MESSAGE_SOURCE / INVALID_MESSAGE_TARGET → fixable: false (user must review connections)
- MISSING_ACTIVATION → fixable: false (layout-dependent, user must place correctly)
- SELF_MESSAGE / MISSING_ALT_FRAGMENT → fixable: false (requires user judgment)

## RESPONSE FORMAT (JSON only, no markdown)
{{
  "errors": [
    {{
      "error_type": "RULE_CODE",
      "severity": "ERROR|WARNING|INFO",
      "element": "LifelineOrMessageName",
      "description": "What is wrong",
      "suggestion": "How to fix",
      "auto_fix": {{
        "fixable": true,
        "action": "add_shape",
        "shape_type": "lifeline",
        "name": "MissingLifelineName"
      }}
    }}
  ],
  "score": 0-100,
  "summary": "e.g. 2 errors, 1 warning"
}}

If correct: {{"errors": [], "score": 100, "summary": "Diagram is correct"}}

## STRICT VALIDATION RULES — MUST FOLLOW:
- Only report issues you are CONFIDENT about. Do NOT invent errors.
- STRICT: If a lifeline is NOT explicitly mentioned in the scenario, do NOT report MISSING_LIFELINE.
- STRICT: If a message/interaction is NOT explicitly mentioned in the scenario, do NOT report MISSING_MESSAGE.
- STRICT: If a return message is NOT explicitly required by scenario, do NOT report MISSING_RETURN.
- STRICT: Do NOT assume messages or lifelines that are implied but not written in scenario.
- STRICT: When in doubt about any error, SKIP it — do not report it.
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
    url = f"{_OPENAI_API_BASE}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 8192,
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
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
        text  = outer["choices"][0]["message"]["content"]
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

def validate_with_openai(
    scenario:     str,
    shapes:       List[Dict[str, Any]],
    diagram_type: str = "class",
) -> Optional[Dict[str, Any]]:
    """
    Validate any diagram type using OpenAI.
    diagram_type: 'class' | 'usecase' | 'sequence'
    Returns structured validation result or None.
    """
    api_key = _get_api_key()
    if not api_key:
        _log.warning("OPENAI_API_KEY not set — skipping AI validation")
        return None

    # ── Sanitize shapes BEFORE building prompt ────────────────────────────────
    # This strips ToolType. prefix and removes cross-diagram shapes so OpenAI
    # doesn't get confused (e.g. actor shapes confusing a class diagram check).
    clean_shapes = _sanitize_shapes(shapes, diagram_type)
    _log.info("Sanitized shapes: %d → %d (diagram: %s)", len(shapes), len(clean_shapes), diagram_type)

    prompt = _build_prompt(diagram_type, scenario, clean_shapes)

    for model in _MODELS:
        _log.info("Trying OpenAI model: %s (diagram: %s)", model, diagram_type)
        result = _call_model(prompt, api_key, model)
        if result:
            _log.info("OpenAI model %s succeeded!", model)
            raw_errors = result.get("errors", [])
            score      = int(result.get("score", 50))
            summary    = result.get("summary", "OpenAI validation complete")

            errors, warnings, info = [], [], []
            for e in raw_errors:
                sev  = str(e.get("severity", "ERROR")).upper()
                # auto_fix: OpenAI se aane wala fix object — Flutter use karega
                raw_fix = e.get("auto_fix", {})
                auto_fix = {
                    "fixable":          bool(raw_fix.get("fixable", False)),
                    "action":           str(raw_fix.get("action",           "")),
                    "shape_type":       str(raw_fix.get("shape_type",       "")),
                    "name":             str(raw_fix.get("name",             "")),
                    "from_element":     str(raw_fix.get("from_element",     "")),
                    "to_element":       str(raw_fix.get("to_element",       "")),
                    "arrow_type":       str(raw_fix.get("arrow_type",       "")),
                    "message_label":    str(raw_fix.get("message_label",    "")),
                    "multiplicity_from":str(raw_fix.get("multiplicity_from","")),
                    "multiplicity_to":  str(raw_fix.get("multiplicity_to",  "")),
                } if raw_fix else {"fixable": False}

                error_type = str(e.get("error_type", "UNKNOWN"))
                element    = str(e.get("element",    ""))

                # ── Fallback: if OpenAI didn't return fixable auto_fix, build one ──
                if not auto_fix.get("fixable"):
                    auto_fix = _build_fallback_fix(error_type, element, e, diagram_type)

                item = {
                    "error_type":  error_type,
                    "severity":    sev,
                    "element":     element,
                    "description": str(e.get("description", "")),
                    "suggestion":  str(e.get("suggestion",  "")),
                    "auto_fix":    auto_fix,
                }
                if sev == "WARNING":   warnings.append(item)
                elif sev == "INFO":    info.append(item)
                else:                  errors.append(item)

            # Count how many errors are auto-fixable
            all_items = errors + warnings + info
            fixable_count = sum(1 for i in all_items if i.get("auto_fix", {}).get("fixable"))

            return {
                "is_valid":      len(errors) == 0,
                "score":         score,
                "summary":       summary,
                "errors":        errors,
                "warnings":      warnings,
                "info":          info,
                "total_issues":  len(raw_errors),
                "fixable_count": fixable_count,
                "source":        "openai",
            }

    _log.error("All OpenAI models failed!")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE-AWARE PROMPT — jab shapes empty ho aur image available ho
# ─────────────────────────────────────────────────────────────────────────────

def _build_image_prompt(diagram_type: str, scenario: str) -> str:
    """
    Jab shapes empty ho aur image available ho — image ko directly analyze karo.
    Shapes ke baghair OpenAI ko image dekhni chahiye taake accurately validate kare.
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

## AUTO-FIX INSTRUCTIONS
For each error, provide an "auto_fix" object. Set "fixable": true only for these:
- MISSING_ACTOR / MISSING_LIFELINE / MISSING_CLASS → action: "add_shape", shape_type, name
- MISSING_USE_CASE → action: "add_shape", shape_type: "use_case_oval", name
- MISSING_SYSTEM_BOUNDARY → action: "add_boundary", shape_type: "system_boundary"
- MISSING_RELATIONSHIP / MISSING_MESSAGE → action: "add_arrow", from_element, to_element, arrow_type, message_label
- MISSING_RETURN → action: "add_arrow", from_element, to_element, arrow_type: "dashed_arrow", message_label
- EMPTY_CLASS_NAME / EMPTY_LIFELINE_NAME → action: "rename_shape", name
- DUPLICATE_* → action: "merge_shapes", name
- DISCONNECTED_ACTOR / ISOLATED_USE_CASE → action: "add_arrow", from_element, to_element, arrow_type: "association"
- MISSING_MULTIPLICITY → action: "add_label", from_element, to_element, multiplicity_from, multiplicity_to
All other errors → fixable: false

## RESPONSE FORMAT (JSON only, no markdown)
{{
  "errors": [
    {{
      "error_type": "RULE_CODE",
      "severity": "ERROR|WARNING|INFO",
      "element": "element name",
      "description": "What is wrong",
      "suggestion": "How to fix",
      "auto_fix": {{
        "fixable": true,
        "action": "add_shape",
        "shape_type": "actor",
        "name": "MissingActorName"
      }}
    }}
  ],
  "score": 0-100,
  "summary": "brief summary"
}}

If correct: {{"errors": [], "score": 100, "summary": "Diagram is correct"}}

## STRICT VALIDATION RULES — MUST FOLLOW:
- Only report issues you are CONFIDENT about from what you SEE. Do NOT invent errors.
- STRICT: If something is NOT explicitly mentioned in the scenario, do NOT report it as missing.
- STRICT: Do NOT report MISSING_ATTRIBUTE or MISSING_METHOD unless scenario explicitly requires them.
- STRICT: Do NOT assume standard attributes (id, name, date, etc.) are required unless scenario says so.
- STRICT: Only report what you can clearly SEE is wrong or missing — when in doubt, SKIP the error.
- STRICT: Check EVERY association arrow for multiplicity labels on BOTH ends. Report MISSING_MULTIPLICITY if any end is missing.

## CLASS NAME vs ATTRIBUTE/METHOD — CRITICAL RULES:
- A UML class box has 3 sections: TOP = class name, MIDDLE = attributes, BOTTOM = methods.
- STRICT: ONLY the TOP section is the class name. Do NOT treat attribute or method names as class names.
- STRICT: Names like 'reviewId', 'staffId', 'patientId', 'staffID' visible in the MIDDLE section are ATTRIBUTES — NOT class names. Never report them as MISSING_CLASS.
- STRICT: Names ending with () like 'submitreview()', 'login()', 'logout()' are METHODS — NOT classes. Never report them as MISSING_CLASS.
- STRICT: Do NOT report SPELLING_MISTAKE if the "corrected" name is an attribute or method visible inside the class box. Example: class named 'Review' with attribute 'reviewId' — do NOT say 'Review' should be 'Reviewid'. This is WRONG.
- STRICT: Do NOT confuse camelCase attribute names (reviewId, staffId) with missing class names."""


def _call_model_with_image(prompt: str, image_b64: str, mime_type: str, api_key: str, model: str) -> Optional[Dict]:
    """OpenAI Vision — image + text prompt dono bhejo."""
    url = f"{_OPENAI_API_BASE}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
            ],
        }],
        "temperature": 0.1,
        "max_tokens": 8192,
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
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
        text  = outer["choices"][0]["message"]["content"]
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


def validate_with_openai_image(
    scenario:     str,
    image_b64:    str,
    mime_type:    str = "image/png",
    diagram_type: str = "class",
) -> Optional[Dict[str, Any]]:
    """
    Image-based validation — shapes ki jagah actual diagram image bhejo OpenAI ko.
    Yeh tab use hota hai jab user gallery se image upload kare (canvas ke baghair).

    diagram_type: 'class' | 'usecase' | 'sequence'
    Returns structured validation result or None.
    """
    api_key = _get_api_key()
    if not api_key:
        _log.warning("OPENAI_API_KEY not set — skipping image validation")
        return None

    # Vision-capable models only
    vision_models = ["gpt-4o"]

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
                raw_fix = e.get("auto_fix", {})
                auto_fix = {
                    "fixable":          bool(raw_fix.get("fixable", False)),
                    "action":           str(raw_fix.get("action",           "")),
                    "shape_type":       str(raw_fix.get("shape_type",       "")),
                    "name":             str(raw_fix.get("name",             "")),
                    "from_element":     str(raw_fix.get("from_element",     "")),
                    "to_element":       str(raw_fix.get("to_element",       "")),
                    "arrow_type":       str(raw_fix.get("arrow_type",       "")),
                    "message_label":    str(raw_fix.get("message_label",    "")),
                    "multiplicity_from":str(raw_fix.get("multiplicity_from","")),
                    "multiplicity_to":  str(raw_fix.get("multiplicity_to",  "")),
                } if raw_fix else {"fixable": False}

                error_type = str(e.get("error_type", "UNKNOWN"))
                element    = str(e.get("element",    ""))

                if not auto_fix.get("fixable"):
                    auto_fix = _build_fallback_fix(error_type, element, e, diagram_type)

                item = {
                    "error_type":  error_type,
                    "severity":    sev,
                    "element":     element,
                    "description": str(e.get("description", "")),
                    "suggestion":  str(e.get("suggestion",  "")),
                    "auto_fix":    auto_fix,
                }
                if sev == "WARNING":  warnings.append(item)
                elif sev == "INFO":   info.append(item)
                else:                 errors.append(item)

            all_items = errors + warnings + info
            fixable_count = sum(1 for i in all_items if i.get("auto_fix", {}).get("fixable"))

            return {
                "is_valid":        len(errors) == 0,
                "score":           score,
                "summary":         summary,
                "errors":          errors,
                "warnings":        warnings,
                "info":            info,
                "total_issues":    len(raw_errors),
                "fixable_count":   fixable_count,
                "source":          "openai-vision",
                "validation_mode": "openai",
            }

    _log.error("All Vision models failed!")
    return None
