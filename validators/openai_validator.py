"""
validators/openai_validator.py
──────────────────────────────────────────────────────────────────────────────
OpenAI AI Validator for ALL diagram types:
  - Class Diagram
  - Use Case Diagram
  - Sequence Diagram

Model: gpt-4o
Rate limit (429): auto wait + retry

FIXES APPLIED:
  1. WRONG_RELATIONSHIP: arrow type mismatch properly detected & reported
  2. WRONG_MULTIPLICITY / WRONG_LABEL: incorrect multiplicity/label reported
  3. WRONG_ASSOCIATION_LABEL: wrong label on association arrow reported
  4. Attributes/methods from scenario → do NOT suggest creating a new class
  5. Errors/suggestions are concise and to-the-point (max 1 sentence each)
  6. Image-based validation: real class/relation/boundary detection fixed
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
# ─────────────────────────────────────────────────────────────────────────────

_TOOLTYPE_MAP = {
    "classfullshape":       "class",
    "classshape":           "class",
    "generalization":       "generalization_arrow",
    "association":          "association_arrow",
    "aggregation":          "aggregation_arrow",
    "composition":          "composition_arrow",
    "dependency":           "dependency_arrow",
    "openarrow":            "association_arrow",    # openArrow = plain association
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
    t = str(raw_type).strip().lower()
    if "." in t:
        t = t.split(".")[-1]
    t = re.sub(r"[_\-]", "", t)
    return _TOOLTYPE_MAP.get(t, t)


def _sanitize_shapes(shapes: List[Dict], diagram_type: str) -> List[Dict]:
    valid_types = _VALID_TYPES_BY_DIAGRAM.get(diagram_type)
    cleaned = []

    for s in shapes:
        clean_t = _clean_type(str(s.get("type", "")))

        if valid_types and clean_t not in valid_types:
            has_conn = any(s.get(f) for f in ("from", "to", "startLifeline", "endLifeline"))
            if not has_conn:
                continue

        entry: Dict[str, Any] = {"type": clean_t}
        for field in ("text", "label", "name", "id", "from", "to",
                      "startLifeline", "endLifeline", "lifelineRef",
                      "arrow_type", "multiplicity_start", "multiplicity_end",
                      "relationship_label", "message_label"):
            val = s.get(field)
            if val is not None and str(val).strip().lower() not in ("", "none", "null", "undefined"):
                if field == "text" and "\n" in str(val):
                    val = str(val).split("\n")[0].strip()
                entry[field] = val
        cleaned.append(entry)

    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK AUTO-FIX BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_fallback_fix(error_type: str, element: str, raw_error: dict, diagram_type: str) -> dict:
    et = error_type.upper()
    desc = str(raw_error.get("description", "")).lower()
    suggestion = str(raw_error.get("suggestion", ""))

    if "MISSING_CLASS" in et:
        return {"fixable": True, "action": "add_shape", "shape_type": "class", "name": element or "NewClass"}
    if "DUPLICATE_CLASS" in et:
        return {"fixable": True, "action": "merge_shapes", "name": element}
    if "EMPTY_CLASS_NAME" in et:
        return {"fixable": True, "action": "rename_shape", "name": element or "ClassName"}
    if "MISSING_RELATIONSHIP" in et:
        from_el, to_el = _parse_from_to(desc + " " + suggestion.lower())
        arrow = _guess_arrow_type(desc + " " + suggestion.lower(), diagram_type)
        if from_el and to_el:
            return {"fixable": True, "action": "add_arrow",
                    "from_element": from_el, "to_element": to_el, "arrow_type": arrow}
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
    if "WRONG_MULTIPLICITY" in et or "MISSING_MULTIPLICITY" in et:
        from_el, to_el = _parse_from_to(desc + " " + suggestion.lower())
        if from_el and to_el:
            return {"fixable": True, "action": "add_label",
                    "from_element": from_el, "to_element": to_el,
                    "multiplicity_from": "1", "multiplicity_to": "*"}
        return {"fixable": False}
    if "WRONG_LABEL" in et or "WRONG_ASSOCIATION_LABEL" in et:
        from_el, to_el = _parse_from_to(desc + " " + suggestion.lower())
        if from_el and to_el:
            return {"fixable": True, "action": "add_label",
                    "from_element": from_el, "to_element": to_el}
        return {"fixable": False}

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
        name = element or "DoAction"
        if suggestion:
            import re as _re
            m = _re.search(r"'([^']+)'", suggestion)
            if m: name = m.group(1)
        return {"fixable": True, "action": "rename_shape", "name": name}

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
    import re as _re
    m = _re.search(r"from ['\"]?([A-Za-z][A-Za-z0-9_\s]*?)['\"]? to ['\"]?([A-Za-z][A-Za-z0-9_\s]*?)['\"]?(?:\s|$|\.)", text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = _re.search(r"between ['\"]?([A-Za-z][A-Za-z0-9_\s]*?)['\"]? and ['\"]?([A-Za-z][A-Za-z0-9_\s]*?)['\"]?(?:\s|$|\.)", text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = _re.search(r"['\"]([A-Za-z][A-Za-z0-9_\s]*?)['\"] class and the ['\"]([A-Za-z][A-Za-z0-9_\s]*?)['\"]", text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None


def _guess_arrow_type(text: str, diagram_type: str) -> str:
    if "composition" in text:   return "composition"
    if "aggregation" in text:   return "aggregation"
    if "generalization" in text or "inherit" in text: return "generalization"
    if "dependency" in text or "depend" in text:      return "dependency"
    if "include" in text:       return "include"
    if "extend" in text:        return "extend"
    return "association"


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS (text-based)
# ─────────────────────────────────────────────────────────────────────────────

def _prompt_class(scenario: str, shapes: List[Dict]) -> str:
    return f"""You are a strict UML Class Diagram validator. Find ALL real errors.

## SCENARIO
{scenario}

## DIAGRAM SHAPES (JSON from student's drawing tool)
{json.dumps(shapes, indent=2)}

## HOW TO READ SHAPE DATA
- "type": "class" = class box; "association_arrow"/"aggregation_arrow"/"composition_arrow"/"generalization_arrow"/"dependency_arrow" = relationship arrows
- "text": for class boxes = full text content (first line = class name; subsequent lines = attributes/methods). NEVER treat attributes or methods as class names.
- "arrow_type": the actual drawn arrow type (e.g. "association", "composition", "generalization")
- "multiplicity_start": multiplicity at the FROM end (e.g. "1", "0..1"). MISSING field = no multiplicity drawn.
- "multiplicity_end": multiplicity at the TO end (e.g. "*", "1..*"). MISSING field = no multiplicity drawn.
- "relationship_label": label on the relationship arrow (e.g. "manages", "teaches")
- "from" / "to": which class the arrow connects

## CRITICAL — BEFORE REPORTING ANY ERROR:
1. Extract from "text" of class shapes: line 0 = class name, remaining lines = attributes/methods.
2. If a word appears as an attribute or method inside a class box → it is NOT a missing class.
3. Only report MISSING_CLASS if the entity has NO class box at all.
4. Attributes/methods listed in scenario belong INSIDE a class box — do NOT report them as missing classes.

## RULES

### R1 — MISSING_CLASS (ERROR)
Every important entity/noun in the scenario needs a class box.
- Check each entity: does a class shape have it as the CLASS NAME (first line of text)?
- If an entity only appears as an attribute inside another class → it is NOT a missing class.

### R2 — EXTRA_CLASS (WARNING)
A class exists that is not mentioned anywhere in the scenario.

### R3 — MISSING_RELATIONSHIP (ERROR)
A relationship stated in the scenario has no arrow drawn between the two classes.

### R4 — WRONG_RELATIONSHIP (ERROR) ← IMPORTANT
The arrow DRAWN between two classes uses the WRONG type vs what the scenario implies.
- Check: arrow's "arrow_type" field vs what the scenario says.
- "consists of" / "part of" / "owns" → must be "composition", not "association"
- "has many" / "collection of" → must be "aggregation", not "association"
- "is a" / "extends" / "type of" → must be "generalization", not "association"
- "uses" / "depends on" → must be "dependency"
- Report: WRONG_RELATIONSHIP with element = "ClassA → ClassB", description = "Used [drawn_type] but scenario implies [correct_type]."

### R5 — WRONG_MULTIPLICITY (WARNING) ← IMPORTANT
An arrow exists and has multiplicity fields, but the VALUES are incorrect for the scenario.
- e.g. Scenario says "one customer places many orders" → must be 1 on Customer end, * on Order end.
- If drawn multiplicity doesn't match → report WRONG_MULTIPLICITY.
- Also report MISSING_MULTIPLICITY if either "multiplicity_start" or "multiplicity_end" field is absent.

### R6 — WRONG_ASSOCIATION_LABEL (WARNING) ← IMPORTANT
An association arrow has a "relationship_label" that does not match what the scenario describes.
- e.g. Scenario says "Teacher teaches Student" but label is "manages" → WRONG_ASSOCIATION_LABEL.
- Also report MISSING_ASSOCIATION_LABEL if scenario mentions a label but none is drawn.

### R7 — WRONG_INHERITANCE (ERROR)
Generalization arrow must go FROM child TO parent. If reversed → error.

### R8 — DUPLICATE_CLASS (ERROR)
Same class name appears more than once.

### R9 — EMPTY_CLASS_NAME (ERROR)
A class box has no name, empty name, or placeholder like "Class1".

### R10 — MISSING_ATTRIBUTE (WARNING)
ONLY if scenario EXPLICITLY names a specific attribute for a class (e.g. "Student has studentId and name").
Do NOT invent attributes. Do NOT report if scenario doesn't specify.

### R11 — MISSING_METHOD (WARNING)
ONLY if scenario EXPLICITLY names a specific method (e.g. "Student can login()").
Do NOT invent methods.

## CONCISENESS RULE
- "description": max 1 sentence, state exactly what is wrong.
- "suggestion": max 1 sentence, state exactly how to fix it.
- Do NOT write paragraphs. Be direct and specific.

## RESPONSE FORMAT (pure JSON, no markdown, no extra text)
{{
  "errors": [
    {{
      "error_type": "RULE_CODE",
      "severity": "ERROR|WARNING|INFO",
      "element": "affected element (e.g. 'Student', 'Student → Order')",
      "description": "One sentence: what is wrong.",
      "suggestion": "One sentence: how to fix it.",
      "auto_fix": {{
        "fixable": true,
        "action": "add_shape|delete_shape|rename_shape|add_arrow|delete_arrow|change_arrow_type|add_label|merge_shapes",
        "shape_type": "class",
        "name": "ClassName",
        "from_element": "SourceClass",
        "to_element": "TargetClass",
        "arrow_type": "association|aggregation|composition|generalization|dependency",
        "multiplicity_from": "1",
        "multiplicity_to": "*"
      }}
    }}
  ],
  "score": 0-100,
  "summary": "X errors, Y warnings."
}}

If fully correct: {{"errors": [], "score": 100, "summary": "Diagram is correct."}}
"""


def _prompt_usecase(scenario: str, shapes: List[Dict]) -> str:
    return f"""You are a strict UML Use Case Diagram validator. Find ALL real errors.

## SCENARIO
{scenario}

## DIAGRAM SHAPES
{json.dumps(shapes, indent=2)}

## HOW TO READ SHAPE DATA
- "type": "actor" = stick figure, "use_case_oval" = oval, "system_boundary" = boundary box
- "text": name of the actor or use case
- "arrow_type": "association" / "include_extend" / "generalization"
- "from" / "to": connected shapes

## RULES

### R1 — MISSING_ACTOR (ERROR)
Every person/external system in the scenario needs an actor shape.

### R2 — EXTRA_ACTOR (WARNING)
Actor in diagram not mentioned in scenario.

### R3 — MISSING_USE_CASE (ERROR)
Every action/function in the scenario needs a use case oval.

### R4 — EXTRA_USE_CASE (INFO)
Use case oval not mentioned in scenario.

### R5 — DISCONNECTED_ACTOR (ERROR)
An actor has no line to any use case.

### R6 — ISOLATED_USE_CASE (ERROR)
A use case has no connection to any actor (directly or via include/extend).

### R7 — MISSING_SYSTEM_BOUNDARY (ERROR)
No system_boundary shape exists in the diagram.

### R8 — WRONG_RELATIONSHIP (ERROR)
include/extend/generalization misused:
- <<include>> = mandatory sub-function
- <<extend>> = optional extension

### R9 — MISSING_VERB_IN_USE_CASE (WARNING)
Use case name has no action verb (e.g. "Password" instead of "Reset Password").

### R10 — DUPLICATE_ACTOR / DUPLICATE_USE_CASE (ERROR)
Same name appears more than once.

## CONCISENESS RULE
- "description": max 1 sentence.
- "suggestion": max 1 sentence.

## RESPONSE FORMAT (pure JSON, no markdown)
{{
  "errors": [
    {{
      "error_type": "RULE_CODE",
      "severity": "ERROR|WARNING|INFO",
      "element": "element name",
      "description": "One sentence: what is wrong.",
      "suggestion": "One sentence: how to fix.",
      "auto_fix": {{
        "fixable": true,
        "action": "add_shape|delete_shape|rename_shape|add_arrow|add_boundary|merge_shapes",
        "shape_type": "actor|use_case_oval|system_boundary",
        "name": "ElementName",
        "from_element": "ActorName",
        "to_element": "UseCaseName",
        "arrow_type": "association|include|extend|generalization"
      }}
    }}
  ],
  "score": 0-100,
  "summary": "X errors, Y warnings."
}}

If correct: {{"errors": [], "score": 100, "summary": "Diagram is correct."}}
"""


def _prompt_sequence(scenario: str, shapes: List[Dict]) -> str:
    return f"""You are a strict UML Sequence Diagram validator. Find ALL real errors.

## SCENARIO
{scenario}

## DIAGRAM SHAPES
{json.dumps(shapes, indent=2)}

## HOW TO READ SHAPE DATA
- "type": "lifeline"/"object_lifeline" = vertical dashed lifeline; "arrow" = solid message; "dashed_arrow" = return message
- "text": lifeline name or message label
- "from" / "startLifeline": sender
- "to" / "endLifeline": receiver
- Shapes appear in draw order (top-to-bottom = earlier index)

## RULES

### R1 — MISSING_LIFELINE (ERROR)
Every participant/object in scenario needs a lifeline.

### R2 — EXTRA_LIFELINE (WARNING)
A lifeline not mentioned in scenario.

### R3 — MISSING_MESSAGE (ERROR)
Every interaction in scenario needs a message arrow.

### R4 — WRONG_MESSAGE_ORDER (ERROR)
Messages not in the chronological order described in scenario.

### R5 — MISSING_RETURN (WARNING)
For synchronous calls, a dashed return arrow should be present if scenario implies a response.

### R6 — ISOLATED_LIFELINE (ERROR)
A lifeline sends AND receives zero messages.

### R7 — EMPTY_LIFELINE_NAME (ERROR)
A lifeline has no name or placeholder name.

### R8 — INVALID_MESSAGE_SOURCE (ERROR)
A message starts from a non-existent lifeline.

### R9 — INVALID_MESSAGE_TARGET (ERROR)
A message targets a non-existent lifeline.

### R10 — MISSING_ALT_FRAGMENT (WARNING)
Scenario has conditional logic but no combined fragment box.

## CONCISENESS RULE
- "description": max 1 sentence.
- "suggestion": max 1 sentence.

## RESPONSE FORMAT (pure JSON, no markdown)
{{
  "errors": [
    {{
      "error_type": "RULE_CODE",
      "severity": "ERROR|WARNING|INFO",
      "element": "element name",
      "description": "One sentence: what is wrong.",
      "suggestion": "One sentence: how to fix.",
      "auto_fix": {{
        "fixable": true,
        "action": "add_shape|rename_shape|add_arrow|merge_shapes",
        "shape_type": "lifeline|activation_box|combined_fragment",
        "name": "ParticipantName",
        "from_element": "SenderLifeline",
        "to_element": "ReceiverLifeline",
        "message_label": "messageLabel",
        "arrow_type": "arrow|dashed_arrow"
      }}
    }}
  ],
  "score": 0-100,
  "summary": "X errors, Y warnings."
}}

If correct: {{"errors": [], "score": 100, "summary": "Diagram is correct."}}
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
        return _prompt_class(scenario, shapes)


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PROMPTS (Vision-based)
# ─────────────────────────────────────────────────────────────────────────────

def _build_image_prompt(diagram_type: str, scenario: str) -> str:
    dt = diagram_type.lower()

    if "usecase" in dt or "use_case" in dt:
        return f"""You are a strict UML Use Case Diagram validator. Analyze this IMAGE carefully.

## SCENARIO
{scenario}

## WHAT TO LOOK FOR IN THE IMAGE

ACTORS: Stick figures (human outline) with a name label below them.
USE CASES: Oval/ellipse shapes with a name label inside them.
SYSTEM BOUNDARY: A rectangle enclosing the use cases (actors are outside).
CONNECTIONS: Lines between actors and ovals; dashed arrows with <<include>> or <<extend>> labels.

## IMPORTANT — READ THE IMAGE CAREFULLY:
- If you see a rectangle around the ovals → system boundary EXISTS, do NOT report it missing.
- Identify EVERY stick figure → that is an actor.
- Identify EVERY oval/ellipse → that is a use case.
- Identify EVERY connection line between them.

## CHECKS

### MISSING_ACTOR (ERROR)
For each person/system in scenario: is a stick figure with that name drawn? If not → error.

### MISSING_USE_CASE (ERROR)
For each action/function in scenario: is an oval with that name drawn? If not → error.

### DISCONNECTED_ACTOR (ERROR)
Any stick figure with NO line to any oval → error.

### ISOLATED_USE_CASE (ERROR)
Any oval with NO connection → error.

### MISSING_SYSTEM_BOUNDARY (ERROR)
ONLY report if there is genuinely NO rectangle anywhere. If a rectangle exists → do NOT report.

### WRONG_RELATIONSHIP (ERROR)
<<include>> or <<extend>> used incorrectly.

### MISSING_VERB_IN_USE_CASE (WARNING)
Oval label has no action verb.

### EXTRA_ACTOR (WARNING)
Stick figure not mentioned in scenario.

### EXTRA_USE_CASE (INFO)
Oval not mentioned in scenario.

## CONCISENESS
- description: 1 sentence max.
- suggestion: 1 sentence max.

## RESPONSE (pure JSON only, no markdown)
{{
  "errors": [
    {{
      "error_type": "RULE_CODE",
      "severity": "ERROR|WARNING|INFO",
      "element": "element name",
      "description": "One sentence.",
      "suggestion": "One sentence.",
      "auto_fix": {{
        "fixable": true,
        "action": "add_shape|rename_shape|add_arrow|add_boundary|merge_shapes",
        "shape_type": "actor|use_case_oval|system_boundary",
        "name": "ElementName",
        "from_element": "ActorName",
        "to_element": "UseCaseName",
        "arrow_type": "association|include|extend|generalization"
      }}
    }}
  ],
  "score": 0-100,
  "summary": "X errors, Y warnings."
}}

If correct: {{"errors": [], "score": 100, "summary": "Diagram is correct."}}
"""

    elif "sequence" in dt:
        return f"""You are a strict UML Sequence Diagram validator. Analyze this IMAGE carefully.

## SCENARIO
{scenario}

## WHAT TO LOOK FOR IN THE IMAGE

LIFELINES: Vertical dashed lines with a name box/label at the top.
MESSAGES: Horizontal arrows between lifelines, with a label on/above the arrow.
RETURN MESSAGES: Dashed horizontal arrows going back.
ACTIVATION BOXES: Narrow rectangles on lifelines showing when a participant is active.

## CHECKS

### MISSING_LIFELINE (ERROR)
For each participant in scenario: is a vertical dashed line with that name drawn? If not → error.

### MISSING_MESSAGE (ERROR)
For each interaction in scenario: is a horizontal arrow with a matching label drawn? If not → error.

### WRONG_MESSAGE_ORDER (ERROR)
Messages not in the top-to-bottom order described in scenario.

### MISSING_RETURN (WARNING)
Synchronous call has no dashed return arrow when scenario implies a response.

### ISOLATED_LIFELINE (ERROR)
A lifeline has no messages at all.

### EMPTY_LIFELINE_NAME (ERROR)
A lifeline has no name label.

### EXTRA_LIFELINE (WARNING)
A lifeline not mentioned in scenario.

## CONCISENESS
- description: 1 sentence max.
- suggestion: 1 sentence max.

## RESPONSE (pure JSON only, no markdown)
{{
  "errors": [
    {{
      "error_type": "RULE_CODE",
      "severity": "ERROR|WARNING|INFO",
      "element": "element name",
      "description": "One sentence.",
      "suggestion": "One sentence.",
      "auto_fix": {{
        "fixable": true,
        "action": "add_shape|rename_shape|add_arrow",
        "shape_type": "lifeline|activation_box|combined_fragment",
        "name": "ParticipantName",
        "from_element": "SenderLifeline",
        "to_element": "ReceiverLifeline",
        "message_label": "messageLabel",
        "arrow_type": "arrow|dashed_arrow"
      }}
    }}
  ],
  "score": 0-100,
  "summary": "X errors, Y warnings."
}}

If correct: {{"errors": [], "score": 100, "summary": "Diagram is correct."}}
"""

    else:  # class diagram
        return f"""You are a strict UML Class Diagram validator. Analyze this IMAGE carefully.

## SCENARIO
{scenario}

## WHAT TO LOOK FOR IN THE IMAGE

CLASS BOXES: Rectangles divided into 3 horizontal sections.
  - TOP section = CLASS NAME (this is the only class name — ignore attributes/methods below)
  - MIDDLE section = attributes (e.g. studentId, name) — these are NOT class names
  - BOTTOM section = methods (e.g. login(), getGrade()) — these are NOT class names

RELATIONSHIP ARROWS (look carefully at the arrowhead style):
  - Solid line + open hollow triangle → GENERALIZATION (inheritance)
  - Solid line + open arrowhead → ASSOCIATION
  - Solid line + filled diamond at source → COMPOSITION
  - Solid line + open diamond at source → AGGREGATION
  - Dashed line + open arrowhead → DEPENDENCY

MULTIPLICITY LABELS: Numbers/symbols near both ends of relationship arrows (e.g. "1", "*", "1..*").
RELATIONSHIP LABELS: Text in the middle of relationship lines (e.g. "teaches", "manages").

## CRITICAL READING RULES
1. The CLASS NAME is ONLY the text in the TOP section of a class box.
2. Text in the MIDDLE or BOTTOM section = attributes/methods — NEVER treat these as missing classes.
3. If a word from the scenario appears as an attribute inside a class box → it is NOT a missing class.
4. Identify the ACTUAL arrowhead style before reporting wrong relationship type.

## CHECKS

### MISSING_CLASS (ERROR)
For each entity in scenario: is a class box with that name (in TOP section) drawn?
- Do NOT report if the entity appears as an attribute inside a class.

### MISSING_RELATIONSHIP (ERROR)
For each relationship in scenario: is the correct arrow drawn between those classes?

### WRONG_RELATIONSHIP (ERROR)
An arrow exists but uses the WRONG type:
- "consists of"/"part of" → must be COMPOSITION (filled diamond), not association
- "is a"/"inherits" → must be GENERALIZATION (hollow triangle), not association
- "has many"/"collection of" → must be AGGREGATION (open diamond)
Report the drawn type and the required type.

### WRONG_MULTIPLICITY (WARNING)
Multiplicity values exist but are incorrect for the scenario (e.g. "1" drawn but should be "*").

### MISSING_MULTIPLICITY (WARNING)
Association/aggregation/composition arrow has no multiplicity label at one or both ends.

### WRONG_ASSOCIATION_LABEL (WARNING)
Arrow has a label that doesn't match what the scenario describes for that relationship.

### WRONG_INHERITANCE (ERROR)
Generalization arrow points FROM parent TO child (should be child → parent with hollow triangle at parent).

### DUPLICATE_CLASS (ERROR)
Same class name in more than one box.

### EMPTY_CLASS_NAME (ERROR)
A class box with no name or placeholder name.

### EXTRA_CLASS (WARNING)
A class box not mentioned in scenario.

## CONCISENESS
- description: 1 sentence max.
- suggestion: 1 sentence max.

## RESPONSE (pure JSON only, no markdown)
{{
  "errors": [
    {{
      "error_type": "RULE_CODE",
      "severity": "ERROR|WARNING|INFO",
      "element": "element name or 'ClassA → ClassB'",
      "description": "One sentence: what is wrong.",
      "suggestion": "One sentence: how to fix.",
      "auto_fix": {{
        "fixable": true,
        "action": "add_shape|delete_shape|rename_shape|add_arrow|change_arrow_type|add_label|merge_shapes",
        "shape_type": "class",
        "name": "ClassName",
        "from_element": "SourceClass",
        "to_element": "TargetClass",
        "arrow_type": "association|aggregation|composition|generalization|dependency",
        "multiplicity_from": "1",
        "multiplicity_to": "*"
      }}
    }}
  ],
  "score": 0-100,
  "summary": "X errors, Y warnings."
}}

If correct: {{"errors": [], "score": 100, "summary": "Diagram is correct."}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# HTTP CALLS
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


def _call_model_with_image(prompt: str, image_b64: str, mime_type: str, api_key: str, model: str) -> Optional[Dict]:
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


# ─────────────────────────────────────────────────────────────────────────────
# RESULT BUILDER (shared logic)
# ─────────────────────────────────────────────────────────────────────────────

def _build_result(result: Dict, diagram_type: str, source: str) -> Dict[str, Any]:
    raw_errors = result.get("errors", [])
    score      = int(result.get("score", 50))
    summary    = result.get("summary", "Validation complete")

    errors, warnings, info = [], [], []
    for e in raw_errors:
        sev  = str(e.get("severity", "ERROR")).upper()
        raw_fix = e.get("auto_fix", {})
        auto_fix = {
            "fixable":           bool(raw_fix.get("fixable", False)),
            "action":            str(raw_fix.get("action",            "")),
            "shape_type":        str(raw_fix.get("shape_type",        "")),
            "name":              str(raw_fix.get("name",              "")),
            "from_element":      str(raw_fix.get("from_element",      "")),
            "to_element":        str(raw_fix.get("to_element",        "")),
            "arrow_type":        str(raw_fix.get("arrow_type",        "")),
            "message_label":     str(raw_fix.get("message_label",     "")),
            "multiplicity_from": str(raw_fix.get("multiplicity_from", "")),
            "multiplicity_to":   str(raw_fix.get("multiplicity_to",   "")),
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
        "source":          source,
        "validation_mode": "openai",
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def validate_with_gemini(
    scenario:     str,
    shapes:       List[Dict[str, Any]],
    diagram_type: str = "class",
) -> Optional[Dict[str, Any]]:
    """Text-based (shapes JSON) validation. Called 'gemini' for backward compat."""
    api_key = _get_api_key()
    if not api_key:
        _log.warning("OPENAI_API_KEY not set — skipping AI validation")
        return None

    clean_shapes = _sanitize_shapes(shapes, diagram_type)
    _log.info("Sanitized shapes: %d → %d (diagram: %s)", len(shapes), len(clean_shapes), diagram_type)

    prompt = _build_prompt(diagram_type, scenario, clean_shapes)

    for model in _MODELS:
        _log.info("Trying OpenAI model: %s (diagram: %s)", model, diagram_type)
        result = _call_model(prompt, api_key, model)
        if result:
            _log.info("OpenAI model %s succeeded!", model)
            return _build_result(result, diagram_type, "openai")

    _log.error("All OpenAI models failed!")
    return None


# Alias used by app.py
validate_with_openai = validate_with_gemini


def validate_with_gemini_image(
    scenario:     str,
    image_b64:    str,
    mime_type:    str = "image/png",
    diagram_type: str = "class",
) -> Optional[Dict[str, Any]]:
    """Vision-based (image upload) validation."""
    api_key = _get_api_key()
    if not api_key:
        _log.warning("OPENAI_API_KEY not set — skipping image validation")
        return None

    vision_models = ["gpt-4o"]
    prompt = _build_image_prompt(diagram_type, scenario)

    for model in vision_models:
        _log.info("Trying Vision model: %s (diagram: %s)", model, diagram_type)
        result = _call_model_with_image(prompt, image_b64, mime_type, api_key, model)
        if result:
            _log.info("Vision model %s succeeded!", model)
            return _build_result(result, diagram_type, "openai-vision")

    _log.error("All Vision models failed!")
    return None


# Alias used by app.py
validate_with_openai_image = validate_with_gemini_image
