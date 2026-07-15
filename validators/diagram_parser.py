"""
validators/diagram_parser.py
─────────────────────────────
Drawing app se aane wale raw shape data ko normalize karta hai.

Problem:
  Har drawing app (Lucidchart, draw.io, custom apps) alag field names use
  karta hai. Validator ko ek consistent format chahiye.

  Raw shape (App se):          Normalized shape (Validator ke liye):
  {                            {
    "shapeType": "Actor",        "type":  "actor",
    "Name": "User",              "label": "User",
    "text": "",                  "id":    "shape-0",
    "id": null,                  "from":  "",        ← arrows ke liye
  }                              "to":    "",
                               }

Supported field name variants:
  Label  : label, text, name, Name, title, content, value, caption
  Type   : type, shapeType, shape_type, kind, elementType, nodeType
  ID     : id, Id, ID, uid, uuid, key, elementId
  From   : from, source, startLifeline, start, sourceId, fromId
  To     : to, target, endLifeline, end, targetId, toId

New rules added (from screenshot analysis):
  - R7 check: object shape empty hone par bhi UNLABELLED error
  - R8 check: arrow ke from/to missing hone par bhi error
  - Deletion symbol (X mark) missing hone par error
  - Actor shape ka label properly read karna
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ── Field name mappings ────────────────────────────────────────────────────────

_LABEL_KEYS = (
    "label", "text", "name", "Name", "title",
    "content", "value", "caption", "displayName",
)

_TYPE_KEYS = (
    "type", "shapeType", "shape_type", "kind",
    "elementType", "nodeType", "shapeKind",
)

_ID_KEYS = (
    "id", "Id", "ID", "uid", "uuid",
    "key", "elementId", "shapeId",
)

_FROM_KEYS = (
    "from", "source", "startLifeline", "start",
    "sourceId", "fromId", "sourceShape", "startId",
)

_TO_KEYS = (
    "to", "target", "endLifeline", "end",
    "targetId", "toId", "targetShape", "endId",
)

# Shape type keywords → normalized category
_LIFELINE_KEYWORDS  = ("lifeline", "actor", "object", "participant", "boundary",
                       "control", "entity", "class",
                       # Flutter ToolType serialization variants
                       "tooltype.actor", "tooltype.object", "tooltype.lifeline")
_MESSAGE_KEYWORDS   = ("arrow", "message", "call", "return", "async", "sync",
                       "signal", "dashed", "dotted", "line", "edge", "connector",
                       "selfmessage", "self_message")
_ACTIVATION_KEYWORDS = ("activation", "executionspecification", "execution",
                        "activationbox", "active", "focus")
_DELETION_KEYWORDS  = ("deletion", "destroy", "x", "cross", "termination",
                       "lifeline_end", "end_marker")

# Combined fragment keywords — alt, opt, loop, par boxes
_FRAGMENT_KEYWORDS  = ("fragment", "combinedfragment", "combined_fragment",
                       "alt", "opt", "loop", "par", "break", "critical",
                       "neg", "ref", "tooltype.fragment")

# Guard condition keywords — [condition] brackets on operands
_GUARD_KEYWORDS     = ("guard", "condition", "operand", "guardcondition",
                       "guard_condition", "interactionoperand")


class DiagramParser:
    """
    Normalize raw drawing-app shapes into a consistent format that the
    validator can reliably read.

    Usage:
        parser  = DiagramParser()
        shapes  = parser.normalize(raw_shapes_from_app)
        # now pass `shapes` to SequenceDiagramValidator.validate()
    """

    def normalize(self, raw_shapes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Raw shapes list lo, normalized shapes list return karo.
        Har shape mein guaranteed fields honge:
          type, label, id, from, to
        """
        normalized = []
        for idx, shape in enumerate(raw_shapes):
            normalized.append(self._normalize_shape(shape, idx))
        return normalized

    def _normalize_shape(self, raw: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Single shape normalize karo."""
        raw_type  = self._extract_first(raw, _TYPE_KEYS,  default="unknown")
        raw_label = self._extract_first(raw, _LABEL_KEYS, default="")
        raw_id    = self._extract_first(raw, _ID_KEYS,    default=None)
        raw_from  = self._extract_first(raw, _FROM_KEYS,  default="")
        raw_to    = self._extract_first(raw, _TO_KEYS,    default="")

        norm_type  = self._normalize_type(raw_type)
        norm_label = self._clean(raw_label)
        norm_id    = self._clean(str(raw_id)) if raw_id is not None else f"index-{idx}"
        norm_from  = self._clean(raw_from)
        norm_to    = self._clean(raw_to)

        # Preserve all original fields + add normalized ones
        result = dict(raw)
        result.update({
            "type":  norm_type,
            "label": norm_label,
            "id":    norm_id,
            "from":  norm_from,
            "to":    norm_to,
            # Keep originals for debugging
            "_raw_type":  raw_type,
            "_raw_label": raw_label,
        })
        return result

    # ── Type normalization ────────────────────────────────────────────────────

    @staticmethod
    def _normalize_type(raw_type: str) -> str:
        """
        Raw type string ko lowercase normalize karo.
        Validator ke keywords isse match karte hain.
        """
        return str(raw_type).strip().lower()

    # ── Shape category helpers ────────────────────────────────────────────────

    @staticmethod
    def is_lifeline(shape: Dict) -> bool:
        t = str(shape.get("type", "")).lower()
        # Pure actors are handled separately by is_actor()
        # Do NOT count them as lifelines to avoid double-counting
        if t in ("tooltype.actor", "actor"):
            return False
        return any(k in t for k in _LIFELINE_KEYWORDS)

    @staticmethod
    def is_actor(shape: Dict) -> bool:
        """Specifically detect actor shapes (Flutter ToolType.actor)."""
        t = str(shape.get("type", "")).lower()
        return "actor" in t

    @staticmethod
    def is_message(shape: Dict) -> bool:
        t = str(shape.get("type", "")).lower()
        return any(k in t for k in _MESSAGE_KEYWORDS)

    @staticmethod
    def is_activation(shape: Dict) -> bool:
        t = str(shape.get("type", "")).lower()
        return any(k in t for k in _ACTIVATION_KEYWORDS)

    @staticmethod
    def is_deletion(shape: Dict) -> bool:
        t = str(shape.get("type", "")).lower()
        return any(k in t for k in _DELETION_KEYWORDS)

    @staticmethod
    def is_fragment(shape: Dict) -> bool:
        """Combined fragment shapes — alt, opt, loop, par boxes."""
        t = str(shape.get("type", "")).lower()
        if any(k in t for k in _FRAGMENT_KEYWORDS):
            return True
        # Also detect by label: a shape labelled "alt"/"opt"/"loop"/"par"
        lbl = str(shape.get("label", "")).strip().lower()
        return lbl in ("alt", "opt", "loop", "par", "break", "critical", "neg", "ref")

    @staticmethod
    def is_guard(shape: Dict) -> bool:
        """Guard condition shapes — [condition] operand markers."""
        t = str(shape.get("type", "")).lower()
        if any(k in t for k in _GUARD_KEYWORDS):
            return True
        # Also detect by label: text wrapped in [ ] brackets
        lbl = str(shape.get("label", "")).strip()
        return lbl.startswith("[") and lbl.endswith("]")

    # ── Extraction helpers ────────────────────────────────────────────────────

    @staticmethod
    def _extract_first(
        d: Dict[str, Any],
        keys: tuple,
        default: Any = "",
    ) -> Any:
        """
        Try multiple key names in order, return first non-None value found.
        Falls back to `default` if nothing found.
        """
        for key in keys:
            val = d.get(key)
            if val is not None:
                return val
        return default

    @staticmethod
    def _clean(value: Any) -> str:
        """
        Convert to string, strip whitespace.
        Return "" for None / "None" / "none" — prevents ghost-label bug.
        """
        s = str(value).strip() if value is not None else ""
        return "" if s.lower() in ("none", "null", "undefined") else s


# ── Standalone parse function (convenience) ───────────────────────────────────

def parse_diagram(raw_shapes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Shortcut function — DiagramParser().normalize() ka wrapper.

    Usage:
        from validators.diagram_parser import parse_diagram
        shapes = parse_diagram(raw_shapes_from_app)
    """
    return DiagramParser().normalize(raw_shapes)
