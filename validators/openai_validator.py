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

# System message — anchors model behaviour; keeps it deterministic + strict
_SYSTEM_MESSAGE = (
    "You are a strict, deterministic UML diagram validator. "
    "You ONLY report errors you are 100% certain about. "
    "You NEVER flag capitalisation/case differences as structural errors. "
    "Case-insensitive matching applies to ALL element names. "
    "You ALWAYS return valid JSON only — no markdown, no prose. "
    "Same input MUST always produce the same output."
)


def _get_api_key() -> Optional[str]:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
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
                # classFullShape text = "ClassName\n---attrs---\n..." — send only class name
                if field == "text" and "\n" in str(val):
                    val = str(val).split("\n")[0].strip()
                entry[field] = val
        cleaned.append(entry)

    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK AUTO-FIX BUILDER
# Jab Gemini fixable: true nahi deta, hum error_type se apna fix banate hain
# ─────────────────────────────────────────────────────────────────────────────

def _build_fallback_fix(error_type: str, element: str, raw_error: dict, diagram_type: str) -> dict:
    """
    Gemini ka auto_fix agar incomplete/missing ho — error_type se fallback fix banao.
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
# RULE-BASED PRE-VALIDATOR
# Shapes se directly check karo — LLM par depend mat karo basic checks ke liye.
# Ye layer 100% deterministic hai. LLM sirf scenario-matching ke liye use hoga.
# ─────────────────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    """Normalize a name for case-insensitive comparison."""
    return str(s).strip().lower()


def _get_names(shapes: List[Dict], *types) -> set:
    """Get normalised names of shapes matching given types."""
    result = set()
    for s in shapes:
        if s.get("type") in types:
            name = s.get("text") or s.get("label") or s.get("name") or ""
            n = _norm(name)
            if n and n not in ("none", "null", "undefined", ""):
                result.add(n)
    return result


def _rule_check_class(shapes: List[Dict]) -> List[Dict]:
    """
    Deterministic rule checks for class diagrams.
    Returns list of error dicts (same format as LLM errors).
    Only checks things we can verify from shape data alone — no scenario needed.
    """
    errors = []
    class_names_raw = {}   # norm_name -> original text
    for s in shapes:
        if s.get("type") == "class":
            text = str(s.get("text") or s.get("label") or s.get("name") or "").strip()
            if not text or _norm(text) in ("none", "null", "undefined", "class 1", "classname"):
                errors.append({
                    "error_type": "EMPTY_CLASS_NAME",
                    "severity": "ERROR",
                    "element": text or "(unnamed)",
                    "description": "Class has no name or has a placeholder name.",
                    "suggestion": "Give this class a meaningful name.",
                    "auto_fix": {"fixable": True, "action": "rename_shape", "name": "NewClass"},
                    "_source": "rule",
                })
            else:
                n = _norm(text)
                if n in class_names_raw:
                    # duplicate
                    errors.append({
                        "error_type": "DUPLICATE_CLASS",
                        "severity": "ERROR",
                        "element": text,
                        "description": f"Class '{text}' appears more than once in the diagram.",
                        "suggestion": f"Remove the duplicate '{text}' class.",
                        "auto_fix": {"fixable": True, "action": "merge_shapes", "name": text},
                        "_source": "rule",
                    })
                else:
                    class_names_raw[n] = text

    # Capitalisation check — WARNING only, not ERROR
    for n, text in class_names_raw.items():
        if text and text[0].islower():
            correct = text[0].upper() + text[1:]
            errors.append({
                "error_type": "WRONG_CLASS_CAPITALISATION",
                "severity": "WARNING",
                "element": text,
                "description": f"Class name '{text}' should start with an uppercase letter.",
                "suggestion": f"Rename '{text}' to '{correct}'.",
                "auto_fix": {"fixable": True, "action": "rename_shape", "name": correct},
                "_source": "rule",
            })

    # Arrow without source/target
    arrow_types = {"association_arrow", "aggregation_arrow", "composition_arrow",
                   "generalization_arrow", "dependency_arrow"}
    for s in shapes:
        if s.get("type") in arrow_types:
            frm = _norm(s.get("from") or "")
            to  = _norm(s.get("to")   or "")
            if not frm or not to:
                errors.append({
                    "error_type": "DISCONNECTED_ARROW",
                    "severity": "WARNING",
                    "element": s.get("type", "arrow"),
                    "description": "A relationship arrow is not connected at both ends.",
                    "suggestion": "Connect both ends of the arrow to class shapes.",
                    "auto_fix": {"fixable": False},
                    "_source": "rule",
                })

    return errors


def _rule_check_usecase(shapes: List[Dict]) -> List[Dict]:
    """Deterministic rule checks for use case diagrams."""
    errors = []

    actor_names   = {}  # norm -> original
    uc_names      = {}  # norm -> original
    has_boundary  = False

    for s in shapes:
        t = s.get("type", "")
        name = str(s.get("text") or s.get("label") or s.get("name") or "").strip()
        n = _norm(name)

        if t == "actor":
            if not n or n in ("none", "null", "undefined", "actor"):
                errors.append({
                    "error_type": "UNLABELLED_ACTOR",
                    "severity": "ERROR",
                    "element": name or "(unnamed)",
                    "description": "An actor has no name.",
                    "suggestion": "Give this actor a meaningful name.",
                    "auto_fix": {"fixable": True, "action": "rename_shape", "name": "Actor"},
                    "_source": "rule",
                })
            else:
                if n in actor_names:
                    errors.append({
                        "error_type": "DUPLICATE_ACTOR",
                        "severity": "ERROR",
                        "element": name,
                        "description": f"Actor '{name}' appears more than once.",
                        "suggestion": f"Remove the duplicate '{name}' actor.",
                        "auto_fix": {"fixable": True, "action": "merge_shapes", "name": name},
                        "_source": "rule",
                    })
                else:
                    actor_names[n] = name

        elif t == "use_case_oval":
            if not n or n in ("none", "null", "undefined", "use case"):
                errors.append({
                    "error_type": "UNLABELLED_USE_CASE",
                    "severity": "ERROR",
                    "element": name or "(unnamed)",
                    "description": "A use case has no label.",
                    "suggestion": "Give this use case a descriptive action name.",
                    "auto_fix": {"fixable": True, "action": "rename_shape", "name": "Use Case"},
                    "_source": "rule",
                })
            else:
                if n in uc_names:
                    errors.append({
                        "error_type": "DUPLICATE_USE_CASE",
                        "severity": "ERROR",
                        "element": name,
                        "description": f"Use case '{name}' appears more than once.",
                        "suggestion": f"Remove the duplicate '{name}' use case.",
                        "auto_fix": {"fixable": True, "action": "merge_shapes", "name": name},
                        "_source": "rule",
                    })
                else:
                    uc_names[n] = name

        elif t == "system_boundary":
            has_boundary = True

    return errors


def _rule_check_sequence(shapes: List[Dict]) -> List[Dict]:
    """Deterministic rule checks for sequence diagrams."""
    errors = []
    lifeline_names = {}  # norm -> original

    for s in shapes:
        t = s.get("type", "")
        if t in ("lifeline", "object_lifeline", "actor"):
            name = str(s.get("text") or s.get("label") or s.get("name") or "").strip()
            n = _norm(name)
            if not n or n in ("none", "null", "undefined"):
                errors.append({
                    "error_type": "UNLABELLED_LIFELINE" if t != "object_lifeline" else "UNLABELLED_OBJECT",
                    "severity": "ERROR",
                    "element": name or "(unnamed)",
                    "description": "A lifeline/participant has no name.",
                    "suggestion": "Give this lifeline a meaningful name.",
                    "auto_fix": {"fixable": True, "action": "rename_shape", "name": "Participant"},
                    "_source": "rule",
                })
            else:
                if n in lifeline_names:
                    errors.append({
                        "error_type": "DUPLICATE_LIFELINE",
                        "severity": "WARNING",
                        "element": name,
                        "description": f"Lifeline '{name}' appears more than once.",
                        "suggestion": f"Remove the duplicate '{name}' lifeline.",
                        "auto_fix": {"fixable": True, "action": "merge_shapes", "name": name},
                        "_source": "rule",
                    })
                else:
                    lifeline_names[n] = name

        elif t in ("arrow", "dashed_arrow", "dotted_arrow", "self_message_arrow"):
            label = str(s.get("label") or s.get("text") or "").strip()
            if t != "self_message_arrow" and (_norm(label) in ("", "none", "null", "undefined")):
                errors.append({
                    "error_type": "UNLABELLED_ARROW",
                    "severity": "WARNING",
                    "element": label or "(unlabelled arrow)",
                    "description": "A message arrow has no label.",
                    "suggestion": "Add a meaningful message name to this arrow.",
                    "auto_fix": {"fixable": True, "action": "rename_shape", "name": "message"},
                    "_source": "rule",
                })

    return errors


def _run_rule_checks(shapes: List[Dict], diagram_type: str) -> List[Dict]:
    """Run the appropriate rule-based checks for the diagram type."""
    dt = diagram_type.lower()
    if "class" in dt:
        return _rule_check_class(shapes)
    elif "usecase" in dt or "use_case" in dt or "use case" in dt:
        return _rule_check_usecase(shapes)
    elif "sequence" in dt:
        return _rule_check_sequence(shapes)
    return []


def _merge_results(rule_errors: List[Dict], llm_result: Dict, diagram_type: str) -> Dict:
    """
    Merge rule-based errors with LLM errors.
    Strategy:
    - Rule errors are TRUSTED — always include them (they are deterministic).
    - LLM errors are FILTERED:
        * Remove any LLM error whose error_type + element already covered by a rule error.
        * Remove capitalisation errors from LLM (we handle those in rules as WARNING).
        * Remove LLM errors that contradict rule findings (e.g. LLM says missing X but rules found X).
    """
    CASE_ERROR_TYPES = {
        "WRONG_CLASS_CAPITALISATION", "WRONG_CAPITALISATION",
        "WRONG_ACTOR_CAPITALISATION", "WRONG_USE_CASE_CAPITALISATION",
    }

    # Build set of (error_type, norm_element) already covered by rules
    rule_covered = {
        (_norm(e["error_type"]), _norm(e["element"]))
        for e in rule_errors
    }

    # Names that EXIST in the diagram (from rule checks — shapes are ground truth)
    existing_names = set()
    for s in llm_result.get("_clean_shapes", []):
        for field in ("text", "label", "name"):
            v = str(s.get(field) or "").strip()
            if v and _norm(v) not in ("none", "null", "undefined", ""):
                existing_names.add(_norm(v))

    filtered_llm_errors = []
    filtered_llm_warnings = []
    filtered_llm_info = []

    all_llm = (
        [(e, "ERROR")   for e in llm_result.get("errors",   [])] +
        [(e, "WARNING") for e in llm_result.get("warnings", [])] +
        [(e, "INFO")    for e in llm_result.get("info",     [])]
    )

    for e, default_sev in all_llm:
        et   = _norm(str(e.get("error_type", "")))
        elem = _norm(str(e.get("element", "")))
        sev  = str(e.get("severity", default_sev)).upper()

        # Drop capitalisation errors — rules handle these as WARNING
        if str(e.get("error_type", "")).upper() in CASE_ERROR_TYPES:
            continue

        # Drop if already covered by a rule error of same type+element
        if (et, elem) in rule_covered:
            continue

        # Drop MISSING_X errors where element actually exists in diagram
        # (LLM hallucinating missing elements that are present)
        if et.startswith("missing_") and elem and elem in existing_names:
            _log.debug("Dropping hallucinated MISSING error: %s for '%s' (element exists)", et, elem)
            continue

        # Keep it
        if sev == "WARNING":
            filtered_llm_warnings.append(e)
        elif sev == "INFO":
            filtered_llm_info.append(e)
        else:
            filtered_llm_errors.append(e)

    # Convert rule errors to final format
    def rule_to_item(r):
        sev = r.get("severity", "ERROR")
        return {
            "error_type":  r["error_type"],
            "severity":    sev,
            "element":     r["element"],
            "description": r["description"],
            "suggestion":  r["suggestion"],
            "auto_fix":    r.get("auto_fix", {"fixable": False}),
        }

    rule_errors_formatted   = [rule_to_item(r) for r in rule_errors if r.get("severity") == "ERROR"]
    rule_warnings_formatted = [rule_to_item(r) for r in rule_errors if r.get("severity") == "WARNING"]
    rule_info_formatted     = [rule_to_item(r) for r in rule_errors if r.get("severity") == "INFO"]

    final_errors   = rule_errors_formatted   + filtered_llm_errors
    final_warnings = rule_warnings_formatted + filtered_llm_warnings
    final_info     = rule_info_formatted     + filtered_llm_info

    all_items = final_errors + final_warnings + final_info
    fixable_count = sum(1 for i in all_items if i.get("auto_fix", {}).get("fixable"))

    score = llm_result.get("score", 100)
    if final_errors:
        # Recalculate score: each ERROR costs more than WARNING
        deduction = len(final_errors) * 15 + len(final_warnings) * 5
        score = max(0, min(100, 100 - deduction))

    summary = llm_result.get("summary", "")
    if not final_errors and not final_warnings and not final_info:
        summary = "Diagram is correct"
        score = 100

    return {
        "is_valid":      len(final_errors) == 0,
        "score":         score,
        "summary":       summary,
        "errors":        final_errors,
        "warnings":      final_warnings,
        "info":          final_info,
        "total_issues":  len(all_items),
        "fixable_count": fixable_count,
        "source":        "openai+rules",
    }


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS — alag diagram type ke liye alag prompt
# ─────────────────────────────────────────────────────────────────────────────

def _prompt_class(scenario: str, shapes: List[Dict]) -> str:
    return f"""You are an expert UML Class Diagram validator using SEMANTIC analysis.

## TASK
This is a CLASS DIAGRAM. Validate it using ONLY class diagram rules. Return ONLY valid JSON.

## ABSOLUTE RESTRICTIONS — violating these makes your response wrong:
- Do NOT check for actors, stick figures, system boundaries, use cases, lifelines, or messages.
- Do NOT report MISSING_ACTOR, MISSING_SYSTEM_BOUNDARY, MISSING_USE_CASE, MISSING_LIFELINE.
- Do NOT apply use case or sequence diagram rules of any kind.
- ONLY apply the rules listed below.

## SCENARIO
{scenario}

## DIAGRAM SHAPES
{json.dumps(shapes, indent=2)}

## HOW TO READ FLUTTER SHAPES — CRITICAL:

### Arrow/Relationship shapes:
- `"type"`: e.g. `"ToolType.association"`, `"ToolType.aggregation"`, `"ToolType.composition"`, `"ToolType.generalization"`, `"ToolType.dependency"`
- `"multiplicity_start"`: Multiplicity at source end e.g. `"1"`, `"0..*"`, `"1..*"`
- `"multiplicity_end"`: Multiplicity at target end e.g. `"*"`, `"1"`, `"0..1"`
- `"relationship_label"`: Label on the arrow e.g. `"manages"`, `"contains"`
- `"from"` / `"to"`: Source and target class names
- `"text"`: Raw text in format `"startMult|label|endMult"` e.g. `"1|manages|*"`

### Class shapes:
- `"type"`: `"ToolType.classShape"` or `"ToolType.classFullShape"`
- `"text"`: Class name (top section)

### MULTIPLICITY VALIDATION RULES:
- If `multiplicity_start` OR `multiplicity_end` field exists with a non-empty value → multiplicity EXISTS.
- If BOTH are null/empty on an association/aggregation/composition arrow → report MISSING_MULTIPLICITY.
- Check `"text"` field too — format is `"startMult|label|endMult"`. If the text has pipe-separated values, those are the multiplicities.
- If multiplicity is present but WRONG vs scenario → report WRONG_MULTIPLICITY with correct expected values.

### RELATIONSHIP TYPE VALIDATION:
- Read `"type"` field: `composition`, `aggregation`, `generalization`, `association`, `dependency`
- If scenario says "is a" / "inherits" / "type of" / "kind of" → expect GENERALIZATION
- If scenario says "consists of" / "composed of" / "cannot exist without" → expect COMPOSITION
- If scenario says "contains" / "collection of" / "holds" / "is made up of" → expect AGGREGATION
- If scenario says "has" / "uses" / "is related to" / "is associated with" → expect ASSOCIATION
- Wrong type drawn → report WRONG_RELATIONSHIP_TYPE

## SEMANTIC ANALYSIS — READ THIS CAREFULLY:
You must use SEMANTIC reasoning, not just keyword matching. Different students describe the same correct diagram in different ways. A diagram is valid if its OVERALL LOGIC matches the scenario's intent, even if exact wording differs.

Examples of semantically equivalent descriptions:
- "Customer places Order" and "Order is placed by Customer" → same relationship
- "Bank manages Accounts" and "Bank has multiple Accounts" → same aggregation
- Multiplicity "1 to many" = "1..*" = "one to many" — all mean the same

## RELATIONSHIP RULES — CRITICAL:
- ONLY report MISSING_RELATIONSHIP if the scenario EXPLICITLY states a relationship between two specific classes using trigger words (has, contains, inherits, etc.).
- If the scenario does NOT describe any relationship between two classes, do NOT invent one.
- Do NOT report relationships that are merely implied or logically reasonable — only what is written.
- A diagram with classes but NO relationships drawn is valid if the scenario does not describe relationships.

## CLASS NAME CASE RULES — CRITICAL:
- ALL class name matching is CASE-INSENSITIVE. "customer" == "Customer" == "CUSTOMER".
- NEVER report MISSING_CLASS or EXTRA_CLASS because of capitalisation differences — they are the SAME class.
- Do NOT report WRONG_CLASS_CAPITALISATION — the rule-based validator handles this separately.
- Focus ONLY on structural errors: wrong relationships, missing relationships per scenario, wrong multiplicity.

## MISSING LABEL RULES:
- If a relationship arrow exists between ClassA and ClassB, and the scenario explicitly names a label for that relationship (e.g. "manages", "contains", "employs"), but the drawn arrow has no label → report MISSING_ASSOCIATION_LABEL.
- Description should say: "The relationship between 'ClassA' and 'ClassB' should have the label 'X' as described in the scenario."
- Suggestion: "Add the label 'X' to the arrow between 'ClassA' and 'ClassB'."
- If the scenario does NOT name a specific label, do NOT report missing label — labels are optional.

## RULES TO CHECK (class diagram ONLY)
1. MISSING_CLASS              — Important nouns in scenario must be classes. Only explicitly named entities.
2. EXTRA_CLASS                — Class in diagram not mentioned in scenario (warning).
3. WRONG_CLASS_CAPITALISATION — Class name exists but starts with lowercase — suggest capitalising, never suggest removing.
4. WRONG_RELATIONSHIP_TYPE    — Wrong arrow type vs scenario (association/aggregation/composition/generalization).
5. MISSING_RELATIONSHIP       — Relationship EXPLICITLY described in scenario but not drawn.
6. MISSING_MULTIPLICITY       — Association/aggregation/composition arrow drawn but both multiplicity fields are empty.
7. WRONG_MULTIPLICITY         — Multiplicity present but value is wrong vs scenario.
8. MISSING_ASSOCIATION_LABEL  — Scenario names a label for a relationship but it is not on the drawn arrow.
9. WRONG_INHERITANCE_DIRECTION — Inheritance arrow reversed (child should point TO parent).
10. DUPLICATE_CLASS           — Same class name appears twice.
11. CIRCULAR_INHERITANCE      — A inherits B and B inherits A.
12. EMPTY_CLASS_NAME          — Class has no name or placeholder like "Class 1".
13. SELF_ASSOCIATION          — Class connected to itself (warn unless scenario says so).

## SEVERITY
ERROR = must fix | WARNING = should fix | INFO = suggestion

## AUTO-FIX INSTRUCTIONS
For each error, provide an "auto_fix" object:
- "action": one of "add_shape", "delete_shape", "rename_shape", "add_arrow", "delete_arrow", "change_arrow_type", "add_label", "merge_shapes"
- "shape_type": (for add_shape) "class"
- "name": new name or label
- "from_element": source class name
- "to_element": target class name
- "arrow_type": "association", "aggregation", "composition", "generalization", "dependency"
- "multiplicity_from": multiplicity at source e.g. "1"
- "multiplicity_to": multiplicity at target e.g. "*"
- "fixable": true/false

AUTO-FIX RULES:
- MISSING_CLASS → fixable: true, action: add_shape, shape_type: class, name: <missing class>
- WRONG_CLASS_CAPITALISATION → fixable: true, action: rename_shape, name: <correctly capitalised name>
- DUPLICATE_CLASS → fixable: true, action: merge_shapes, name: <class name to keep>
- EMPTY_CLASS_NAME → fixable: true, action: rename_shape, name: <correct name from scenario>
- WRONG_RELATIONSHIP_TYPE → fixable: true, action: change_arrow_type, from_element, to_element, arrow_type: <correct type>
- MISSING_RELATIONSHIP → fixable: true, action: add_arrow, from_element, to_element, arrow_type: <correct type>
- MISSING_MULTIPLICITY → fixable: true, action: add_label, from_element, to_element, multiplicity_from, multiplicity_to
- WRONG_MULTIPLICITY → fixable: true, action: add_label, from_element, to_element, multiplicity_from: <correct>, multiplicity_to: <correct>
- MISSING_ASSOCIATION_LABEL → fixable: true, action: add_label, from_element, to_element, name: <label>
- WRONG_INHERITANCE_DIRECTION → fixable: false
- CIRCULAR_INHERITANCE → fixable: false
- EXTRA_CLASS → fixable: false
- SELF_ASSOCIATION → fixable: false

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

## STRICT RULES — MUST FOLLOW:
- Results must be DETERMINISTIC — same diagram + scenario must always give same errors.
- Only report issues you are CONFIDENT about. Do NOT invent errors.
- ALL name matching is CASE-INSENSITIVE — never flag a name as wrong just because of capitalisation.
- The rule-based validator already checks: empty names, duplicates, disconnected arrows, capitalisation.
  Your job is ONLY: scenario-based missing elements, wrong relationship types, wrong multiplicities.
- Before including ANY error, ask: "Can I see this is wrong in the shapes list?" — if not, SKIP it.
- A diagram with correct structure but no scenario violations → return empty errors, score 100.

### MISSING_CLASS:
- STRICT: Only report for nouns EXPLICITLY written as class names in the scenario.
- STRICT: Method names like "submitOrder()" are METHODS, never classes.
- STRICT: Attribute names like "orderId", "price" are ATTRIBUTES, never classes.

### MISSING_RELATIONSHIP:
- STRICT: Only report if scenario EXPLICITLY uses trigger words: has, contains, inherits, is a type of, consists of, is composed of, manages, holds, etc.
- STRICT: Do NOT invent relationships just because two classes exist in the same scenario.
- STRICT: If scenario says nothing about the relationship between two classes, no MISSING_RELATIONSHIP error.

### CLASS CAPITALISATION:
- STRICT: If a class "customer" exists and scenario mentions "Customer" → WRONG_CLASS_CAPITALISATION only. Do NOT say remove it. Do NOT say add "Customer" as a new class.

### MISSING_MULTIPLICITY:
- STRICT: Only report if an association/aggregation/composition arrow is drawn AND both multiplicity_start and multiplicity_end are empty/missing.
- STRICT: Check the "text" field for pipe format "startMult|label|endMult" before concluding multiplicity is missing.

### MISSING_ASSOCIATION_LABEL:
- STRICT: Only report if scenario EXPLICITLY mentions what the relationship should be called (e.g. "Bank manages Customer" → label is "manages").
- STRICT: If scenario does not name the relationship, do NOT report missing label.

### General:
- STRICT: When in doubt about ANY error, SKIP it.
- STRICT: Use semantic understanding — a diagram correct in logic is correct even if wording differs.
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
5. DISCONNECTED_ACTOR    — Actor has NO line connecting it to ANY use case. A line that touches any part of the actor stick-figure (head, body, hands, or feet) counts as connected. Only flag DISCONNECTED_ACTOR if there is literally no line endpoint near the actor at all.
6. ISOLATED_USE_CASE     — Use case has no connection to any actor.
7. MISSING_SYSTEM_BOUNDARY — System boundary box is missing entirely.
8. WRONG_SYSTEM_BOUNDARY_NAME — System boundary EXISTS but its label does not match the system name in the scenario. Do NOT say "add a new boundary" — say "change the name from X to Y".
9. WRONG_RELATIONSHIP    — include/extend/generalization used incorrectly.
10. MISSING_VERB_IN_USE_CASE — Use case name missing action verb (e.g. "Payment" instead of "Make Payment").
11. DUPLICATE_ACTOR      — Same actor name appears twice.
12. DUPLICATE_USE_CASE   — Same use case name appears twice.
13. ACTOR_NOT_IN_BOUNDARY — Use cases should be inside system boundary.

## CASE-INSENSITIVE MATCHING — CRITICAL
- ALL name matching is CASE-INSENSITIVE. "Login" == "login" == "LOGIN".
- NEVER report MISSING_ACTOR, MISSING_USE_CASE, or EXTRA_* just because of capitalisation.
- Do NOT report WRONG_CAPITALISATION — the rule-based validator handles this separately.
- Your job: check if scenario-required actors/use-cases are present (case-insensitive) and connected.

## ACTOR CONNECTION — CRITICAL RULES
- An actor stick-figure occupies vertical space: head at top, body in middle, hands on sides, feet at bottom.
- A line endpoint touching ANY part of the actor (head, body, hands, feet area) = CONNECTED.
- Only report DISCONNECTED_ACTOR if no line whatsoever is near the actor.
- NEVER report DISCONNECTED_ACTOR just because a line touches the body/torso instead of the head.

## SYSTEM BOUNDARY NAME RULES — CRITICAL
- If a system boundary rectangle EXISTS with a label that does not match the scenario system name:
  → Report WRONG_SYSTEM_BOUNDARY_NAME.
  → Description: "System boundary is named 'X' but scenario calls it 'Y'."
  → Suggestion: "Change the system boundary name from 'X' to 'Y'." (never say "add a new boundary")
- If system boundary does NOT exist at all → report MISSING_SYSTEM_BOUNDARY.
- If system boundary exists with correct name → no error.

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
- WRONG_SYSTEM_BOUNDARY_NAME → fixable: true, action: rename_shape, name: <correct system name from scenario>
- DISCONNECTED_ACTOR → fixable: true, action: add_arrow, from_element: <actor name>, to_element: <use case name>, arrow_type: association
- ISOLATED_USE_CASE → fixable: true, action: add_arrow, from_element: <actor name>, to_element: <use case name>, arrow_type: association
- MISSING_VERB_IN_USE_CASE → fixable: true, action: rename_shape, name: <corrected name with verb>
- WRONG_CAPITALISATION → fixable: true, action: rename_shape, name: <correctly capitalised name>
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
- ALL name matching is CASE-INSENSITIVE — never flag a name as missing/wrong due to capitalisation.
- The rule-based validator already checks: empty names, duplicates, unlabelled shapes.
  Your job is ONLY: scenario-based missing actors/use-cases, disconnected elements, wrong relationships.
- Before including ANY error, ask: "Is this element genuinely missing from the shapes list (case-insensitive)?" — if not, SKIP.
- A diagram with correct structure but no scenario violations → return empty errors, score 100.

### MISSING_ACTOR hallucination prevention:
- STRICT: Only report MISSING_ACTOR for persons/systems that are EXPLICITLY written in the scenario.
- STRICT: Do NOT invent actors that are implied but not written.
- STRICT: A single actor is enough if scenario mentions only one person/system.

### MISSING_USE_CASE hallucination prevention:
- STRICT: Only report MISSING_USE_CASE for actions that are EXPLICITLY written in the scenario.
- STRICT: Do NOT split one use case into multiple — if scenario says "login", do NOT also require "validate credentials", "check password" etc.
- STRICT: Do NOT invent sub-use-cases that are not written in the scenario.
- STRICT: Matching is CASE-INSENSITIVE — "Login" and "login" are the same use case.

### DISCONNECTED_ACTOR hallucination prevention:
- STRICT: A line touching the actor's body, torso, or any limb area = CONNECTED. Do NOT report DISCONNECTED_ACTOR for such actors.
- STRICT: Only report DISCONNECTED_ACTOR if there is genuinely no line anywhere near the actor shape.

### WRONG_SYSTEM_BOUNDARY_NAME rules:
- STRICT: If the boundary EXISTS with a wrong name → say "Change the name from 'X' to 'Y'". Never say "add a new boundary".
- STRICT: Do NOT report both MISSING_SYSTEM_BOUNDARY and WRONG_SYSTEM_BOUNDARY_NAME for the same diagram.

### WRONG_RELATIONSHIP hallucination prevention:
- STRICT: Only report WRONG_RELATIONSHIP if you are 100% certain the relationship type is wrong.
- STRICT: Do NOT flag association arrows as wrong unless scenario explicitly requires include/extend.

### General:
- STRICT: When in doubt about ANY error, SKIP it — do not report it.
- STRICT: Do NOT assume actors or use cases that are implied but not written in scenario.
- STRICT: Results must be DETERMINISTIC — same diagram + scenario must always produce the same errors.
"""


def _prompt_sequence(scenario: str, shapes: List[Dict]) -> str:
    return f"""You are an expert UML Sequence Diagram validator using SEMANTIC analysis.

## TASK
This is a SEQUENCE DIAGRAM. Validate it using ONLY sequence diagram rules. Return ONLY valid JSON.

## ABSOLUTE RESTRICTIONS:
- Do NOT check for class boxes, system boundaries, or use cases.
- Do NOT report MISSING_CLASS, MISSING_SYSTEM_BOUNDARY, MISSING_USE_CASE.
- Do NOT apply class diagram or use case diagram rules of any kind.

## SCENARIO
{scenario}

## DIAGRAM SHAPES
{json.dumps(shapes, indent=2)}

## HOW TO READ SEQUENCE DIAGRAM SHAPES:

### Lifeline / Object shapes:
- Types: `lifeline`, `object_lifeline`, `object` → these are participant boxes at the top.
- `"text"` or `"label"` field = the participant's name.
- An `object` shape is an OBJECT (instance), not an anonymous lifeline. If it has a name, use that name — do NOT report it as unnamed.
- If an object shape has an empty label → report UNLABELLED_OBJECT (not UNLABELLED_LIFELINE).

### Actor shapes:
- Type: `actor` → stick figure at top, represents a human participant.
- `"text"` or `"label"` = actor's name.

### Message arrows:
- Types: `arrow`, `dashed_arrow`, `dotted_arrow` → horizontal arrows between lifelines.
- `"from"` / `"to"` = source and target lifeline names.
- `"label"` or `"text"` = the message/operation name.
- `"type": "self_message_arrow"` or `"selfmessagearrow"` → self-message (same lifeline sends to itself). This is VALID in UML — do NOT report it as an error unless it has no label.

### Deletion / Destroy markers:
- Types: `deletion_marker`, `deletion`, `destroy`, `x`, `cross` → X mark at bottom of a lifeline.
- `"lifeline"`, `"on"`, or `"label"` field links deletion to a lifeline.
- If these shapes exist in the diagram → deletion symbols ARE present for those lifelines.
- Report MISSING_DELETION_SYMBOL only for lifelines that have NO associated deletion shape.

### Activation boxes:
- Types: `activation_box`, `activation` → thin rectangle on a lifeline's dashed line.

### Combined fragments:
- Types: `combined_fragment`, `fragment` → alt/opt/loop/par boxes.

## SEMANTIC ANALYSIS — READ THIS CAREFULLY:
You must use SEMANTIC reasoning. Different students may label messages differently but mean the same thing. A diagram is correct if its overall interaction logic matches the scenario's intent.

Examples:
- "validateUser()" and "validate credentials" both represent the same login validation step → semantically the same.
- "loginResponse" and "authToken returned" both represent the login result → same.
- Message order is correct if the LOGICAL sequence matches the scenario, even if exact wording differs.

## MESSAGE ORDER RULES — CRITICAL:
- Only report WRONG_MESSAGE_ORDER if the sequence in the diagram is CLEARLY and DEFINITIVELY wrong compared to the scenario's described flow.
- Do NOT report WRONG_MESSAGE_ORDER if the order could be interpreted as valid or if the scenario doesn't specify strict ordering.
- When in doubt about message order → DO NOT report it. Skip the error.
- Semantic equivalents count as correct order (a "validate" step before a "response" step is standard flow).

## SELF-MESSAGE RULES:
- `self_message_arrow` shapes are VALID UML — they represent a method call on the same object.
- Do NOT report SELF_MESSAGE as an error unless the scenario specifically says self-messages are wrong.
- Only report INVALID_SELF_MESSAGE if a self-message arrow has no label at all.

## DELETION SYMBOL RULES — CRITICAL:
- Look for shapes of type `deletion_marker`, `deletion`, `destroy`, `x`, `cross`, or any X-shaped marker.
- If deletion shapes are present in the diagram shapes list → deletion symbols EXIST. Do NOT report them as missing.
- Only report MISSING_DELETION_SYMBOL for specific lifelines that have no associated deletion shape.
- If the diagram uses a simple format (no deletion markers) → report NO_DELETION_SYMBOLS as INFO only (not ERROR).

## OBJECT vs LIFELINE NAMING:
- An `object_lifeline` or `object` shape with a name IS a named participant — do NOT report it as having no name.
- If an object shape's label is empty → report: "Object has no name. Add a name to the object box."
- Do NOT say "lifeline has no name" when the shape type is `object` or `object_lifeline` — say "object has no name".

## RULES TO CHECK (sequence diagram ONLY)
1. MISSING_LIFELINE         — Every participant/object in scenario must have a lifeline or object box.
2. EXTRA_LIFELINE           — Lifeline not in scenario (warning).
3. MISSING_MESSAGE          — Important interaction in scenario not shown as a message arrow. Use semantic matching.
4. WRONG_MESSAGE_ORDER      — Messages are in CLEARLY wrong order vs scenario. Only report when certain.
5. MISSING_RETURN           — A call message has no return/response when scenario explicitly expects one.
6. INVALID_MESSAGE_SOURCE   — Message arrow starts from non-existent lifeline.
7. INVALID_MESSAGE_TARGET   — Message arrow ends at non-existent lifeline.
8. ISOLATED_LIFELINE        — Lifeline sends/receives no messages.
9. UNLABELLED_LIFELINE      — Lifeline box has no label.
10. UNLABELLED_OBJECT       — Object box has no label (use "object" terminology, not "lifeline").
11. MISSING_ACTIVATION      — Lifeline that receives messages has no activation box.
12. MISSING_DELETION_SYMBOL — Specific lifeline has no X/destroy marker at its end.
13. UNLABELLED_ARROW        — Message arrow has no label/name.
14. MISSING_ALT_FRAGMENT    — Conditional logic in scenario not shown as alt/opt fragment.

## SEVERITY
ERROR = must fix | WARNING = should fix | INFO = suggestion

## AUTO-FIX INSTRUCTIONS
auto_fix fields:
- "action": one of "add_shape", "rename_shape", "add_arrow", "merge_shapes"
- "shape_type": one of "lifeline", "activation_box", "combined_fragment", "deletion_marker"
- "name": label to set
- "from_element": source lifeline name
- "to_element": target lifeline name
- "message_label": label for the arrow
- "arrow_type": "arrow" (solid call) or "dashed_arrow" (return/response)
- "fixable": true/false

AUTO-FIX RULES:
- MISSING_LIFELINE → fixable: true, action: add_shape, shape_type: lifeline, name: <participant name>
- UNLABELLED_LIFELINE / UNLABELLED_OBJECT → fixable: true, action: rename_shape, name: <correct name>
- UNLABELLED_ARROW → fixable: true, action: rename_shape, name: <message label from scenario>
- MISSING_MESSAGE → fixable: true, action: add_arrow, from_element, to_element, message_label, arrow_type: arrow
- MISSING_RETURN → fixable: true, action: add_arrow, from_element: <receiver>, to_element: <sender>, message_label: <return label>, arrow_type: dashed_arrow
- MISSING_DELETION_SYMBOL → fixable: true, action: add_shape, shape_type: deletion_marker, name: <lifeline name>
- WRONG_MESSAGE_ORDER → fixable: false
- ISOLATED_LIFELINE → fixable: false
- EXTRA_LIFELINE → fixable: false
- INVALID_MESSAGE_SOURCE / INVALID_MESSAGE_TARGET → fixable: false
- MISSING_ACTIVATION → fixable: false
- MISSING_ALT_FRAGMENT → fixable: false

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

## STRICT RULES — MUST FOLLOW:
- Results must be DETERMINISTIC — same diagram + scenario must always give the same errors.
- Only report issues you are CONFIDENT about. Do NOT invent errors.
- ALL name matching is CASE-INSENSITIVE — never flag a name as missing/wrong due to capitalisation.
- The rule-based validator already checks: empty names, duplicates, unlabelled arrows.
  Your job is ONLY: scenario-based missing lifelines/messages, wrong message order, wrong structure.
- Before including ANY error, ask: "Is this genuinely wrong per the scenario?" — if not, SKIP.
- A diagram with correct structure but no scenario violations → return empty errors, score 100.

### MISSING_LIFELINE:
- STRICT: Only report for participants EXPLICITLY named in the scenario.
- STRICT: An `object` shape with a name counts as a valid lifeline — do NOT report it missing.

### MISSING_MESSAGE:
- STRICT: Only for interactions EXPLICITLY described in the scenario.
- STRICT: Use semantic matching — "validateUser()" matches "validate credentials" — same interaction.
- STRICT: Do NOT invent intermediate messages not in scenario.

### WRONG_MESSAGE_ORDER:
- STRICT: Only report if the order is CLEARLY wrong (e.g. response comes before request).
- STRICT: If the order is ambiguous or could be valid → DO NOT report it. Skip.
- STRICT: Semantic equivalents count as correct — do not flag stylistic differences as wrong order.

### SELF-MESSAGE:
- STRICT: `self_message_arrow` is VALID UML. Do NOT report it as an error.
- STRICT: Only flag if it has absolutely no label.

### DELETION SYMBOLS:
- STRICT: If deletion_marker/deletion/destroy/x shapes are in the diagram → they ARE present. Do NOT report them missing.
- STRICT: Only report MISSING_DELETION_SYMBOL for lifelines that have genuinely no X marker.

### OBJECT vs LIFELINE:
- STRICT: An `object` or `object_lifeline` shape is an object box, not a "lifeline". Use correct terminology.
- STRICT: If an object has a name, it is named — do NOT say it has no name.

### MISSING_RETURN:
- STRICT: Only report if scenario explicitly says there should be a response.
- STRICT: Many valid sequence diagrams have no return arrows.

### General:
- STRICT: When in doubt about ANY error, SKIP it.
- STRICT: Use semantic understanding — diagrams correct in logic are correct.
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
        "messages": [
            {"role": "system", "content": _SYSTEM_MESSAGE},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0,    # deterministic — prevents errors changing on re-validation
        "seed": 42,          # OpenAI seed for extra determinism
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

    # Step 1: Sanitize shapes (strip ToolType prefix, filter cross-diagram shapes)
    clean_shapes = _sanitize_shapes(shapes, diagram_type)
    _log.info("Sanitized shapes: %d → %d (diagram: %s)", len(shapes), len(clean_shapes), diagram_type)

    # Step 2: Run deterministic rule-based checks first
    # These are 100% reliable — no LLM variance, no hallucinations
    rule_errors = _run_rule_checks(clean_shapes, diagram_type)
    _log.info("Rule-based checks found %d issues", len(rule_errors))

    # Step 3: Ask LLM ONLY for scenario-semantic validation
    # LLM checks: are the right classes/actors/lifelines present per scenario?
    # Are relationships semantically correct per scenario description?
    prompt = _build_prompt(diagram_type, scenario, clean_shapes)

    for model in _MODELS:
        _log.info("Trying OpenAI model: %s (diagram: %s)", model, diagram_type)
        llm_result = _call_model(prompt, api_key, model)
        if llm_result:
            _log.info("OpenAI model %s succeeded!", model)

            # Normalize LLM errors into errors/warnings/info lists
            raw_errors = llm_result.get("errors", [])
            llm_errors_list, llm_warnings_list, llm_info_list = [], [], []
            for e in raw_errors:
                sev = str(e.get("severity", "ERROR")).upper()
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
                if sev == "WARNING": llm_warnings_list.append(item)
                elif sev == "INFO":  llm_info_list.append(item)
                else:                llm_errors_list.append(item)

            # Step 4: Merge rule results + filtered LLM results
            normalized_llm = {
                "errors":        llm_errors_list,
                "warnings":      llm_warnings_list,
                "info":          llm_info_list,
                "score":         int(llm_result.get("score", 50)),
                "summary":       llm_result.get("summary", ""),
                "_clean_shapes": clean_shapes,   # passed to merge for hallucination filter
            }
            return _merge_results(rule_errors, normalized_llm, diagram_type)

    # LLM failed — return rule-based results only (still useful)
    _log.error("All OpenAI models failed — returning rule-based results only")
    errors_only   = [r for r in rule_errors if r.get("severity") == "ERROR"]
    warnings_only = [r for r in rule_errors if r.get("severity") == "WARNING"]
    info_only     = [r for r in rule_errors if r.get("severity") == "INFO"]
    all_items     = errors_only + warnings_only + info_only
    return {
        "is_valid":      len(errors_only) == 0,
        "score":         max(0, 100 - len(errors_only) * 15 - len(warnings_only) * 5),
        "summary":       "Validated using rule-based checks (AI unavailable)",
        "errors":        [_format_rule_item(r) for r in errors_only],
        "warnings":      [_format_rule_item(r) for r in warnings_only],
        "info":          [_format_rule_item(r) for r in info_only],
        "total_issues":  len(all_items),
        "fixable_count": sum(1 for r in all_items if r.get("auto_fix", {}).get("fixable")),
        "source":        "rules-only",
    }


def _format_rule_item(r: Dict) -> Dict:
    return {
        "error_type":  r["error_type"],
        "severity":    r.get("severity", "ERROR"),
        "element":     r["element"],
        "description": r["description"],
        "suggestion":  r["suggestion"],
        "auto_fix":    r.get("auto_fix", {"fixable": False}),
    }


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE-AWARE PROMPT — jab shapes empty ho aur image available ho
# ─────────────────────────────────────────────────────────────────────────────

def _build_image_prompt(diagram_type: str, scenario: str) -> str:
    """
    Jab shapes empty ho aur image available ho — image ko directly analyze karo.
    """
    dt = diagram_type.lower()

    if "usecase" in dt or "use_case" in dt:
        rules = """1. MISSING_ACTOR — Every person/system in scenario must be an actor (stick figure).
2. EXTRA_ACTOR — Actor not mentioned in scenario (warning).
3. MISSING_USE_CASE — Every action/function in scenario must be a use case oval.
4. EXTRA_USE_CASE — Use case oval not in scenario (info).
5. DISCONNECTED_ACTOR — Actor has NO line to any use case. A line touching ANY part of the actor (head, body, hands, feet) = CONNECTED. Only flag if truly no line exists near the actor.
6. ISOLATED_USE_CASE — Use case has no connection to any actor.
7. MISSING_SYSTEM_BOUNDARY — System boundary box is entirely absent. Only report if you CANNOT SEE any rectangle/box enclosing use cases.
8. WRONG_SYSTEM_BOUNDARY_NAME — Boundary EXISTS but its label doesn't match scenario system name. Say "change name from X to Y" — never say "add a new boundary".
9. WRONG_RELATIONSHIP — include/extend/generalization used incorrectly.
10. MISSING_VERB_IN_USE_CASE — Use case name missing action verb.
11. DUPLICATE_ACTOR — Same actor name appears twice.
12. DUPLICATE_USE_CASE — Same use case name appears twice.
CASE-INSENSITIVE: "Login" and "login" are the same — do NOT flag capitalisation as missing/extra."""
        dtype_label = "USE CASE"
        extra_rules = """
## ACTOR CONNECTION RULE:
- An actor stick-figure occupies vertical space (head, body, hands, feet).
- Any line touching ANY part of the stick-figure = CONNECTED.
- Only flag DISCONNECTED_ACTOR if there is genuinely no line anywhere near the actor.

## SYSTEM BOUNDARY NAME RULE:
- If boundary EXISTS with wrong name → report WRONG_SYSTEM_BOUNDARY_NAME, suggest renaming.
- If boundary does NOT exist → report MISSING_SYSTEM_BOUNDARY.
- Never report both for the same diagram."""

    elif "sequence" in dt:
        rules = """1. MISSING_LIFELINE — Every participant in scenario must have a lifeline or object box.
2. MISSING_MESSAGE — Important interaction in scenario not shown. Use SEMANTIC matching — equivalent messages count.
3. WRONG_MESSAGE_ORDER — ONLY report if order is CLEARLY and DEFINITIVELY wrong. Skip if any doubt.
4. MISSING_RETURN — Only if scenario explicitly requires a response message.
5. ISOLATED_LIFELINE — Lifeline sends/receives no messages.
6. UNLABELLED_LIFELINE — Lifeline box has no label.
7. UNLABELLED_OBJECT — Object box has no label (say "object" not "lifeline").
8. MISSING_DELETION_SYMBOL — Lifeline has no X/destroy marker. Only report if X symbols are genuinely absent.
9. UNLABELLED_ARROW — Message arrow has no label."""
        dtype_label = "SEQUENCE"
        extra_rules = """
## SELF-MESSAGE RULE:
- Self-message arrows (looping back to same lifeline) are VALID UML — do NOT report them as errors.

## DELETION SYMBOL RULE:
- If you SEE any X marks at lifeline ends → deletion symbols ARE present. Do NOT report them missing.
- Only report MISSING_DELETION_SYMBOL for lifelines with genuinely no X at the bottom.

## OBJECT vs LIFELINE:
- An object box (rectangle with name) IS a named participant. Do NOT report it as unnamed.
- If the object is truly empty/unlabelled → say "Object has no name", not "lifeline has no name".

## MESSAGE ORDER:
- Only report WRONG_MESSAGE_ORDER if you are 100% certain. When in doubt → SKIP."""

    else:
        rules = """1. MISSING_CLASS — Important nouns in scenario must be class boxes. Only explicitly named entities.
2. EXTRA_CLASS — Class not in scenario (warning).
3. WRONG_CLASS_CAPITALISATION — Class starts with lowercase. Suggest capitalising. Do NOT say remove it.
4. WRONG_RELATIONSHIP_TYPE — Wrong arrow type used.
5. MISSING_RELATIONSHIP — Relationship EXPLICITLY in scenario but not drawn. Do NOT invent relationships.
6. MISSING_MULTIPLICITY — Association arrow drawn but multiplicity labels are absent.
7. WRONG_MULTIPLICITY — Multiplicity present but value differs from scenario.
8. MISSING_ASSOCIATION_LABEL — Scenario names a label for a relationship but it's not on the arrow.
9. EMPTY_CLASS_NAME — Class has no name or placeholder like "Class 1"."""
        dtype_label = "CLASS"
        extra_rules = """
## CLASS CAPITALISATION RULE:
- If a class "customer" exists and scenario has "Customer" → report WRONG_CLASS_CAPITALISATION.
- Suggestion: "Capitalise the first letter: rename 'customer' to 'Customer'."
- Do NOT report it as EXTRA_CLASS or say to remove it.

## RELATIONSHIP RULE:
- Only report MISSING_RELATIONSHIP if scenario EXPLICITLY states a relationship (has, contains, inherits, etc.).
- Do NOT invent relationships just because two classes exist.

## MISSING LABEL RULE:
- Only report MISSING_ASSOCIATION_LABEL if scenario names what the relationship should be called.
- If scenario does not name the relationship → labels are optional."""

    return f"""You are an expert UML {dtype_label} Diagram validator using SEMANTIC analysis. You are given an IMAGE of the diagram.

## CRITICAL INSTRUCTION
Look carefully at the ACTUAL IMAGE provided. Validate ONLY what you can SEE.
- Only report something as MISSING if it is genuinely absent from the image.
- Do NOT invent errors for things that are present but styled differently than expected.
- Results must be DETERMINISTIC — same image + scenario must always produce the same errors.

## SCENARIO
{scenario}

## RULES TO CHECK ({dtype_label} diagram ONLY)
{rules}
{extra_rules}

## SEMANTIC ANALYSIS:
- Use semantic reasoning — a diagram correct in overall logic is correct even if exact wording differs.
- Semantically equivalent labels count as correct (e.g. "validateUser()" ≈ "validate credentials").

## SEVERITY
ERROR = must fix | WARNING = should fix | INFO = suggestion

## AUTO-FIX INSTRUCTIONS
For each error, provide an "auto_fix" object. Set "fixable": true only for these:
- MISSING_ACTOR / MISSING_LIFELINE / MISSING_CLASS → action: "add_shape", shape_type, name
- MISSING_USE_CASE → action: "add_shape", shape_type: "use_case_oval", name
- MISSING_SYSTEM_BOUNDARY → action: "add_boundary", shape_type: "system_boundary"
- WRONG_SYSTEM_BOUNDARY_NAME → action: "rename_shape", name: <correct name>
- WRONG_CLASS_CAPITALISATION → action: "rename_shape", name: <capitalised name>
- MISSING_RELATIONSHIP / MISSING_MESSAGE → action: "add_arrow", from_element, to_element, arrow_type, message_label
- MISSING_RETURN → action: "add_arrow", from_element, to_element, arrow_type: "dashed_arrow", message_label
- MISSING_DELETION_SYMBOL → action: "add_shape", shape_type: "deletion_marker", name: <lifeline name>
- EMPTY_CLASS_NAME / UNLABELLED_LIFELINE / UNLABELLED_OBJECT → action: "rename_shape", name
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

## STRICT RULES — MUST FOLLOW:
- Only report issues you are CONFIDENT about from what you SEE. Do NOT invent errors.
- STRICT: Same diagram + scenario must always produce same results (deterministic).
- STRICT: If something is NOT explicitly mentioned in the scenario, do NOT report it as missing.
- STRICT: Method names ending with () are METHODS not class names.
- STRICT: Attribute names are ATTRIBUTES not class names.
- STRICT: Do NOT report MISSING_ATTRIBUTE or MISSING_METHOD unless scenario explicitly requires them.
- STRICT: Do NOT invent sub-use-cases, intermediate messages, or implied relationships not in scenario.
- STRICT: Only report what you can clearly SEE is wrong — when in doubt, SKIP the error.
- STRICT: Use semantic matching — equivalent labels count as correct."""


def _call_model_with_image(prompt: str, image_b64: str, mime_type: str, api_key: str, model: str) -> Optional[Dict]:
    """OpenAI Vision — image + text prompt dono bhejo."""
    url = f"{_OPENAI_API_BASE}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                ],
            },
        ],
        "temperature": 0,    # deterministic — prevents errors changing on re-validation
        "seed": 42,          # OpenAI seed for extra determinism
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
                "validation_mode": "gemini",
            }

    _log.error("All Vision models failed!")
    return None
