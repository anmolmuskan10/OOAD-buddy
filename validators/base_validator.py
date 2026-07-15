"""
validators/base_validator.py
─────────────────────────────
Har validator is base class ko inherit karta hai.
Common helper methods yahan hain.

Updated: ValidationError now supports optional 'source' field
         so hybrid results can tag rule-based vs AI findings.
"""

from typing import List, Dict, Any


class ValidationError:
    """Single validation issue."""

    SEVERITY_ERROR   = "ERROR"
    SEVERITY_WARNING = "WARNING"
    SEVERITY_INFO    = "INFO"

    SOURCE_RULE    = "rule"
    SOURCE_AI      = "ai"
    SOURCE_UNKNOWN = ""

    def __init__(
        self,
        error_type:  str,
        description: str,
        suggestion:  str,
        severity:    str = "ERROR",
        element:     str = "",
        source:      str = SOURCE_RULE,   # NEW: tag which engine found this
    ):
        self.error_type  = error_type
        self.description = description
        self.suggestion  = suggestion
        self.severity    = severity
        self.element     = element
        self.source      = source

    def to_dict(self) -> Dict[str, str]:
        return {
            "error_type":  self.error_type,
            "severity":    self.severity,
            "element":     self.element,
            "description": self.description,
            "suggestion":  self.suggestion,
            "source":      self.source,
        }


class BaseValidator:
    """Abstract base for all diagram validators."""

    def validate(
        self,
        extracted: Dict[str, Any],
        shapes:    List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        raise NotImplementedError

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        return text.strip().lower()

    @staticmethod
    def _get_shape_labels(shapes: List[Dict], shape_type_key: str) -> List[str]:
        """
        shapes list se specific type ke labels nikaalo.
        shape_type_key example: 'actor', 'useCase', 'class'
        """
        labels = []
        for s in shapes:
            shape_type = str(s.get("type", "")).lower()
            if shape_type_key.lower() in shape_type:
                text = str(s.get("text", "")).strip()
                if text:
                    labels.append(text)
        return labels

    @staticmethod
    def _get_all_labels(shapes: List[Dict]) -> List[str]:
        """All text labels from all shapes."""
        return [
            str(s.get("text", "")).strip()
            for s in shapes
            if str(s.get("text", "")).strip()
        ]

    @staticmethod
    def _fuzzy_match(word: str, label_list: List[str]) -> bool:
        """
        Case-insensitive partial match.
        e.g. 'Admin' matches 'admin', 'AdminUser', 'system_admin'
        """
        word_lower = word.lower()
        for label in label_list:
            if word_lower in label.lower() or label.lower() in word_lower:
                return True
        return False

    @staticmethod
    def _build_result(
        errors:  List[ValidationError],
        score:   int,
        summary: str,
    ) -> Dict[str, Any]:
        return {
            "is_valid":     len([e for e in errors if e.severity == "ERROR"]) == 0,
            "score":        score,
            "summary":      summary,
            "errors":       [e.to_dict() for e in errors if e.severity == "ERROR"],
            "warnings":     [e.to_dict() for e in errors if e.severity == "WARNING"],
            "info":         [e.to_dict() for e in errors if e.severity == "INFO"],
            "total_issues": len(errors),
        }
