"""
validators/openai_validator.py
──────────────────────────────────────────────────────────────────────────────
OpenAI AI Validator for ALL diagram types:
  - Class Diagram
  - Use Case Diagram
  - Sequence Diagram

Model: gpt-4o
Rate limit (429): auto wait + retry

IMPROVED: Better prompts for all 3 diagram types.
  - Scenario se entities/actors/classes extract karo
  - Har cheez check karo jo scenario mein hai
  - False positives avoid karo
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
                      "startLifeline", "endLifeline", "lifelineRef"):
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
    if "MISSING_MULTIPLICITY" in et:
        from_el, to_el = _parse_from_to(desc + " " + suggestion.lower())
        if from_el and to_el:
            return {"fixable": True, "action": "add_label",
                    "from_element": from_el, "to_element": to_el,
                    "multiplicity_from": "1", "multiplicity_to": "*"}
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
# IMPROVED PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _prompt_class(scenario: str, shapes: List[Dict]) -> str:
    return f"""You are a strict UML Class Diagram validator. Your job is to find ALL real errors.

## YOUR TASK
1. Read the SCENARIO carefully — identify every important NOUN (these should be classes).
2. Read the DIAGRAM SHAPES — these are what the student actually drew.
3. Compare them: report every mismatch, missing element, and wrong relationship.

## SCENARIO
{scenario}

## DIAGRAM SHAPES (what student drew)
{json.dumps(shapes, indent=2)}

## HOW TO READ SHAPE DATA
Each shape has these important fields:
- "type": shape type (ToolType.classFullShape = class box, ToolType.association = association arrow, etc.)
- "text": for class boxes = class name; for arrows = raw "startMult|label|endMult" string
- "arrow_type": READABLE arrow type — "association", "aggregation", "composition", "generalization", "dependency"
- "multiplicity_start": multiplicity at the FROM end (e.g. "1", "0..1")
- "multiplicity_end": multiplicity at the TO end (e.g. "*", "1..*")
- "relationship_label": label on the arrow (e.g. "manages", "contains")
- "from" / "to": which class the arrow connects

IMPORTANT FOR MULTIPLICITY CHECK:
- If an association/aggregation/composition arrow does NOT have "multiplicity_start" field → start end is MISSING multiplicity
- If an association/aggregation/composition arrow does NOT have "multiplicity_end" field → end end is MISSING multiplicity
- Both ends MUST have multiplicity → report MISSING_MULTIPLICITY if either is absent

## RULES TO CHECK

### R1 — MISSING_CLASS (ERROR)
Every important noun/entity in the scenario MUST appear as a class shape.
- Extract all key nouns from the scenario.
- For each noun: if no class shape has a matching name → MISSING_CLASS error.
- Match names case-insensitively and allow minor spelling variations.

### R2 — EXTRA_CLASS (WARNING)
A class exists in diagram but is NOT mentioned anywhere in the scenario.
- Only report if the class name has NO relation to any scenario concept.

### R3 — MISSING_RELATIONSHIP (ERROR)
If scenario explicitly states a relationship between two entities, it must be drawn.
- "has", "contains", "owns", "manages", "consists of" → association or composition
- "is a", "extends", "inherits" → generalization arrow
- "uses", "depends on" → dependency
- For each stated relationship: if no arrow exists between those classes → error.

### R4 — WRONG_RELATIONSHIP (ERROR)
An arrow exists but the relationship TYPE is wrong vs what scenario implies.
- "consists of" / "part of" / "owns" → should be composition, not association
- "has many" / "collection of" → should be aggregation
- "is a" / "type of" → should be generalization, not association

### R5 — MISSING_MULTIPLICITY (WARNING)
Every association arrow MUST have multiplicity labels on BOTH ends (e.g. 1, *, 1..*, 0..1).
- Check every arrow of type: association_arrow, aggregation_arrow, composition_arrow.
- If EITHER end is missing a multiplicity label → MISSING_MULTIPLICITY warning.

### R6 — WRONG_INHERITANCE (ERROR)
Generalization arrow must point FROM child TO parent.
- If arrow goes from parent to child → WRONG_INHERITANCE error.

### R7 — DUPLICATE_CLASS (ERROR)
Same class name appears more than once in shapes.

### R8 — EMPTY_CLASS_NAME (ERROR)
A class shape has no name, empty name, or placeholder like "Class1", "NewClass".

### R9 — CIRCULAR_INHERITANCE (ERROR)
Class A inherits from B AND B inherits from A → circular.

### R10 — MISSING_ATTRIBUTE (WARNING)
ONLY report if scenario EXPLICITLY mentions a specific attribute for a class.
Example: "Each Student has a studentId and name" → Student must have studentId, name.
Do NOT assume standard attributes. Do NOT report if scenario doesn't specify.

### R11 — MISSING_METHOD (WARNING)
ONLY report if scenario EXPLICITLY mentions a specific operation/method for a class.
Example: "Student can login() and logout()" → Student must have login(), logout().
Do NOT assume methods. Do NOT report if scenario doesn't specify.

## CRITICAL RULES TO AVOID FALSE POSITIVES
- A class box has 3 sections: TOP = class name, MIDDLE = attributes, BOTTOM = methods.
- NEVER treat attribute names (camelCase like studentId, orderId) as class names.
- NEVER treat method names (ending with ()) as class names.
- NEVER report SPELLING_MISTAKE if the "corrected" name is actually an attribute visible inside the box.
- Only report an error when you are CERTAIN it is a real mistake.

## RESPONSE FORMAT (JSON only, no markdown fences)
{{
  "errors": [
    {{
      "error_type": "RULE_CODE",
      "severity": "ERROR|WARNING|INFO",
      "element": "affected element name",
      "description": "Clear explanation of what is wrong",
      "suggestion": "Exactly how to fix it",
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
  "summary": "X errors, Y warnings found"
}}

If diagram is fully correct: {{"errors": [], "score": 100, "summary": "Diagram is correct"}}
"""


def _prompt_usecase(scenario: str, shapes: List[Dict]) -> str:
    return f"""You are a strict UML Use Case Diagram validator. Your job is to find ALL real errors.

## YOUR TASK
1. Read the SCENARIO carefully.
   - Identify every PERSON or EXTERNAL SYSTEM mentioned → these are ACTORS (stick figures).
   - Identify every ACTION, FUNCTION, or FEATURE mentioned → these are USE CASES (ovals).
2. Read the DIAGRAM SHAPES — these are what the student actually drew.
3. Compare them: report every mismatch, missing element, and wrong connection.

## SCENARIO
{scenario}

## DIAGRAM SHAPES (what student drew)
{json.dumps(shapes, indent=2)}

## HOW TO READ SHAPE DATA
- "type": ToolType.actor = stick figure actor, ToolType.useCase = oval, ToolType.systemBoundary = boundary box
- "text": name of the actor/use case
- "arrow_type": "association" = actor-usecase line, "include_extend" = include/extend arrow, "generalization" = inheritance
- "from" / "to": which shapes the arrow connects

## RULES TO CHECK

### R1 — MISSING_ACTOR (ERROR)
Every person or external system in the scenario MUST appear as an actor shape.
- List all people/systems from scenario. Check each one in shapes.
- If an actor from scenario is not drawn → MISSING_ACTOR error.

### R2 — EXTRA_ACTOR (WARNING)
An actor exists in diagram but is not mentioned in scenario.
- Only report if clearly not related to any scenario concept.

### R3 — MISSING_USE_CASE (ERROR)
Every action/function/feature in the scenario MUST appear as a use case oval.
- List all actions from scenario. Check each one in shapes.
- If a use case from scenario is not drawn → MISSING_USE_CASE error.

### R4 — EXTRA_USE_CASE (INFO)
A use case oval exists but is not mentioned in scenario.

### R5 — DISCONNECTED_ACTOR (ERROR)
An actor exists but has NO association line to ANY use case.
- Every actor must be connected to at least one use case.

### R6 — ISOLATED_USE_CASE (ERROR)
A use case oval exists but has NO connection to ANY actor (directly or via include/extend).
- Every use case must be reachable from at least one actor.

### R7 — MISSING_SYSTEM_BOUNDARY (ERROR)
The system boundary rectangle is missing entirely.
- Look for a shape of type "system_boundary" in the shapes list.
- If NO system_boundary shape exists → report this error.
- If a system_boundary shape exists → do NOT report this error.

### R8 — WRONG_RELATIONSHIP (ERROR)
include/extend/generalization used incorrectly:
- <<include>> = base use case ALWAYS calls included use case (mandatory).
- <<extend>> = extension adds optional behavior to base use case.
- Generalization between actors = one actor is a specialized version of another.
- Report if these are reversed or misused.

### R9 — MISSING_VERB_IN_USE_CASE (WARNING)
Use case names must start with or contain an action verb.
- Bad: "Login Page", "Password" → Good: "Login", "Reset Password"
- Report if a use case name has no verb.

### R10 — DUPLICATE_ACTOR (ERROR)
Same actor name appears more than once.

### R11 — DUPLICATE_USE_CASE (ERROR)
Same use case name appears more than once.

### R12 — ACTOR_INSIDE_BOUNDARY (WARNING)
Actors should be OUTSIDE the system boundary box, not inside it.

## CRITICAL RULES TO AVOID FALSE POSITIVES
- Do NOT report MISSING_ACTOR for roles that are only implied, not explicitly stated.
- Do NOT report MISSING_USE_CASE for actions not mentioned in scenario.
- Match names flexibly: "Place Order" matches "placeOrder" or "order placement".
- Only report when you are CERTAIN it is a real mistake.

## RESPONSE FORMAT (JSON only, no markdown fences)
{{
  "errors": [
    {{
      "error_type": "RULE_CODE",
      "severity": "ERROR|WARNING|INFO",
      "element": "affected element name",
      "description": "Clear explanation of what is wrong",
      "suggestion": "Exactly how to fix it",
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
  "summary": "X errors, Y warnings found"
}}

If diagram is fully correct: {{"errors": [], "score": 100, "summary": "Diagram is correct"}}
"""


def _prompt_sequence(scenario: str, shapes: List[Dict]) -> str:
    return f"""You are a strict UML Sequence Diagram validator. Your job is to find ALL real errors.

## YOUR TASK
1. Read the SCENARIO carefully.
   - Identify every PARTICIPANT/OBJECT/SYSTEM mentioned → these need lifelines.
   - Identify every INTERACTION/MESSAGE/CALL described → these need message arrows.
   - Note the ORDER of interactions → sequence matters.
2. Read the DIAGRAM SHAPES — these are what the student actually drew.
3. Compare them: report every mismatch, missing element, and wrong sequence.

## SCENARIO
{scenario}

## DIAGRAM SHAPES (what student drew)
{json.dumps(shapes, indent=2)}

## HOW TO READ SHAPE DATA
- "type": ToolType.lifeLine = lifeline, ToolType.arrow = solid message, ToolType.dashedArrow = return message
- "text": for lifelines = participant name; for arrows = message label
- "message_label": label on message arrow (same as text for arrows)
- "arrow_type": "message_arrow" = solid call, "dashed_arrow" = return/response, "self_message" = self-call
- "from" / "startLifeline": sender lifeline name
- "to" / "endLifeline": receiver lifeline name

MESSAGE ORDER: Shapes appear in the JSON list in the order they were drawn.
Top-to-bottom in diagram = earlier index in this list.

## RULES TO CHECK

### R1 — MISSING_LIFELINE (ERROR)
Every participant, object, or system mentioned in scenario MUST have a lifeline.
- List all participants from scenario. Check each in shapes (lifeline or object_lifeline types).
- If a participant is missing → MISSING_LIFELINE error.

### R2 — EXTRA_LIFELINE (WARNING)
A lifeline exists that is not mentioned in scenario.

### R3 — MISSING_MESSAGE (ERROR)
Every significant interaction described in scenario MUST be shown as a message arrow.
- List all interactions/calls from scenario. Check each in shapes (arrow types).
- If an interaction is missing → MISSING_MESSAGE error.
- Arrows are identified by their "text" or "label" field and by "from"/"to" lifelines.

### R4 — WRONG_MESSAGE_ORDER (ERROR)
Messages must appear in the correct chronological order as described in scenario.
- If scenario says "A calls B, then B calls C" — check the vertical order of arrows.
- If order is reversed → WRONG_MESSAGE_ORDER error.

### R5 — MISSING_RETURN (WARNING)
For every synchronous call (solid arrow), there should be a return (dashed arrow) back.
- If scenario explicitly describes a response/return → it MUST be drawn.
- If scenario does not mention a return → only warn if it's clearly a request-response pattern.

### R6 — ISOLATED_LIFELINE (ERROR)
A lifeline exists but sends AND receives NO messages at all.
- Every lifeline must participate in at least one message.

### R7 — EMPTY_LIFELINE_NAME (ERROR)
A lifeline shape has no name or empty/placeholder name.

### R8 — INVALID_MESSAGE_SOURCE (ERROR)
A message arrow starts from a lifeline that doesn't exist in the diagram.

### R9 — INVALID_MESSAGE_TARGET (ERROR)
A message arrow ends at a lifeline that doesn't exist in the diagram.

### R10 — MISSING_ACTIVATION (WARNING)
A lifeline that receives messages should have an activation box (activation_box shape).
- Only report if activation boxes are used for other lifelines but missing for this one.
- Do NOT report if no activation boxes are used anywhere in diagram.

### R11 — MISSING_ALT_FRAGMENT (WARNING)
If scenario describes conditional logic ("if", "when", "otherwise", "alternatively") →
a combined_fragment (alt/opt) should be present.
- Only report if the condition is clearly stated and important to the flow.

## CRITICAL RULES TO AVOID FALSE POSITIVES
- Match lifeline names flexibly (case-insensitive, partial match ok).
- Do NOT report MISSING_RETURN unless scenario explicitly requires a response.
- Do NOT assume interactions not mentioned in scenario.
- Only report when you are CERTAIN it is a real mistake.

## RESPONSE FORMAT (JSON only, no markdown fences)
{{
  "errors": [
    {{
      "error_type": "RULE_CODE",
      "severity": "ERROR|WARNING|INFO",
      "element": "affected element name",
      "description": "Clear explanation of what is wrong",
      "suggestion": "Exactly how to fix it",
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
  "summary": "X errors, Y warnings found"
}}

If diagram is fully correct: {{"errors": [], "score": 100, "summary": "Diagram is correct"}}
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
# IMPROVED IMAGE PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

def _build_image_prompt(diagram_type: str, scenario: str) -> str:
    dt = diagram_type.lower()

    if "usecase" in dt or "use_case" in dt:
        return f"""You are a strict UML Use Case Diagram validator. You are given an IMAGE of the diagram.

## YOUR TASK
1. Read the SCENARIO carefully.
   - List every PERSON/SYSTEM mentioned → these are required actors (stick figures).
   - List every ACTION/FUNCTION mentioned → these are required use cases (ovals).
2. Look carefully at the IMAGE — identify all drawn elements.
3. Compare and report every mismatch.

## SCENARIO
{scenario}

## WHAT TO CHECK IN THE IMAGE

### MISSING_ACTOR (ERROR)
For each person/system in scenario: is a stick figure drawn with that name? If not → error.

### MISSING_USE_CASE (ERROR)
For each action/function in scenario: is an oval drawn with that name? If not → error.

### DISCONNECTED_ACTOR (ERROR)
Is any stick figure drawn with NO line connecting to any oval? If yes → error.

### ISOLATED_USE_CASE (ERROR)
Is any oval drawn with NO connection to any actor? If yes → error.

### MISSING_SYSTEM_BOUNDARY (ERROR)
Is there a rectangle/box enclosing the use cases?
- If you can clearly see a rectangle → do NOT report this.
- If there is genuinely NO rectangle anywhere → report MISSING_SYSTEM_BOUNDARY.

### WRONG_RELATIONSHIP (ERROR)
Are <<include>> or <<extend>> labels used correctly?
- <<include>> = mandatory sub-function
- <<extend>> = optional extension

### MISSING_VERB_IN_USE_CASE (WARNING)
Do any oval labels lack an action verb? (e.g., "Password" instead of "Reset Password")

### DUPLICATE_ACTOR or DUPLICATE_USE_CASE (ERROR)
Are any actor or use case names repeated?

### EXTRA_ACTOR (WARNING)
Is any stick figure drawn that is NOT mentioned in scenario?

### EXTRA_USE_CASE (INFO)
Is any oval drawn that is NOT mentioned in scenario?

## SEVERITY
ERROR = must fix | WARNING = should fix | INFO = suggestion

## RESPONSE FORMAT (JSON only, no markdown fences)
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
  "summary": "X errors, Y warnings found"
}}

If correct: {{"errors": [], "score": 100, "summary": "Diagram is correct"}}
"""

    elif "sequence" in dt:
        return f"""You are a strict UML Sequence Diagram validator. You are given an IMAGE of the diagram.

## YOUR TASK
1. Read the SCENARIO carefully.
   - List every PARTICIPANT/OBJECT mentioned → these need vertical dashed lifelines.
   - List every INTERACTION/MESSAGE described → these need horizontal arrows.
   - Note the ORDER of interactions.
2. Look carefully at the IMAGE — identify all drawn elements.
3. Compare and report every mismatch.

## SCENARIO
{scenario}

## WHAT TO CHECK IN THE IMAGE

### MISSING_LIFELINE (ERROR)
For each participant in scenario: is a vertical dashed line drawn with that name? If not → error.

### MISSING_MESSAGE (ERROR)
For each interaction in scenario: is a horizontal arrow drawn for it? If not → error.
Check arrow labels — they should match the interaction names in scenario.

### WRONG_MESSAGE_ORDER (ERROR)
Are the message arrows in the correct top-to-bottom order as described in scenario?
If scenario says "A then B then C" but diagram shows different order → error.

### MISSING_RETURN (WARNING)
For request-response interactions: is there a dashed return arrow back?
Only report if scenario clearly implies a response.

### ISOLATED_LIFELINE (ERROR)
Is any lifeline drawn that sends AND receives zero messages?

### EMPTY_LIFELINE_NAME (ERROR)
Is any lifeline drawn without a name label?

### MISSING_ALT_FRAGMENT (WARNING)
If scenario has conditional logic (if/else, optional steps): is there a combined fragment box?

### EXTRA_LIFELINE (WARNING)
Is any lifeline drawn that is NOT mentioned in scenario?

## SEVERITY
ERROR = must fix | WARNING = should fix | INFO = suggestion

## RESPONSE FORMAT (JSON only, no markdown fences)
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
  "summary": "X errors, Y warnings found"
}}

If correct: {{"errors": [], "score": 100, "summary": "Diagram is correct"}}
"""

    else:  # class diagram
        return f"""You are a strict UML Class Diagram validator. You are given an IMAGE of the diagram.

## YOUR TASK
1. Read the SCENARIO carefully.
   - List every important NOUN/ENTITY → these need class boxes.
   - Note every RELATIONSHIP described between entities.
2. Look carefully at the IMAGE — identify all drawn class boxes, arrows, labels.
3. Compare and report every mismatch.

## SCENARIO
{scenario}

## WHAT TO CHECK IN THE IMAGE

### MISSING_CLASS (ERROR)
For each important noun/entity in scenario: is a class box drawn with that name?
- Class boxes have 3 sections: top=name, middle=attributes, bottom=methods.
- The CLASS NAME is only the text in the TOP section.
- Do NOT treat attribute names (camelCase like studentId) as class names.
- Do NOT treat method names (ending with ()) as class names.

### MISSING_RELATIONSHIP (ERROR)
For each relationship described in scenario: is the correct arrow drawn?
- "has/contains/owns" → association or composition arrow
- "is a / inherits" → generalization arrow (hollow triangle head)
- "uses/depends" → dashed dependency arrow

### WRONG_RELATIONSHIP (ERROR)
Is the drawn arrow type correct for the scenario?
- "part of / consists of" → must be composition (filled diamond), not plain association
- "is a" → must be generalization (hollow arrow), not association

### MISSING_MULTIPLICITY (WARNING)
Do all association/aggregation/composition arrows have multiplicity labels on BOTH ends?
Look for numbers like "1", "*", "1..*", "0..1" near each end of every relationship arrow.
If any end is missing a multiplicity label → report.

### WRONG_INHERITANCE (ERROR)
Generalization arrow must point FROM child TO parent (hollow triangle at parent end).
If reversed → error.

### DUPLICATE_CLASS (ERROR)
Same class name appears more than once.

### EMPTY_CLASS_NAME (ERROR)
Any class box with no name or placeholder name like "Class1".

### EXTRA_CLASS (WARNING)
A class box exists but is not mentioned in scenario at all.

## SEVERITY
ERROR = must fix | WARNING = should fix | INFO = suggestion

## RESPONSE FORMAT (JSON only, no markdown fences)
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
  "summary": "X errors, Y warnings found"
}}

If correct: {{"errors": [], "score": 100, "summary": "Diagram is correct"}}
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

def validate_with_openai(
    scenario:     str,
    shapes:       List[Dict[str, Any]],
    diagram_type: str = "class",
) -> Optional[Dict[str, Any]]:
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


def validate_with_openai_image(
    scenario:     str,
    image_b64:    str,
    mime_type:    str = "image/png",
    diagram_type: str = "class",
) -> Optional[Dict[str, Any]]:
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
