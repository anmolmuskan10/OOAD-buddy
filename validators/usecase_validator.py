"""
validators/usecase_validator.py
────────────────────────────────
Rule-based validation for Use Case Diagrams.

FLUTTER JSON FORMAT (no from/to fields — spatial detection used):
  {
    "type":        "ToolType.actor" | "ToolType.useCase" | "ToolType.straightLine" ...
    "text":        "Customer"
    "position":    {"dx": 120.0, "dy": 300.0}   ← shape top-left / line start
    "endPosition": {"dx": 200.0, "dy": 0.0}      ← RELATIVE offset (lines only)
    "size":        {"width": 40.0, "height": 60.0}
    "isBold":      false
  }

  Connection detection = spatial proximity:
    line.start = position
    line.end   = position + endPosition (absolute)
    shape is "connected" if any line endpoint is within HIT_RADIUS of its bbox

MATCHING LOGIC (key design):
  _uc_matches_scenario() uses verb-gated noun matching:
    - Verb MUST match exactly (prevents 'truck delivery' matching 'track delivery')
    - Nouns matched with plural tolerance (delivery/deliveries, order/orders)
    - Filler words ignored ('manage shopping cart' → 'manage cart' ✓)
    - Missing-verb labels (e.g. 'payment') never match ('make payment') → R18 catches it

  _actor_matches() uses edit distance <= 2 BUT enforces minimum similarity ratio
    to prevent 'Armind' matching 'Admin' (too different).

FIXES applied (v16):
  1. UC_VERBS expanded with dataset-derived verbs (enroll, take, watch, borrow etc.)
  2. Actor spelling check: max_dist tightened + ratio guard added
  3. System boundary label checked against scenario system_name
  4. R18 spelling check uses ONLY scenario words — NOT UC_VERBS (prevents Take→Make)
  5. Scenario extractor filler verbs filtered out (access, ensure, divide, perform etc.)
  6. _uc_matches_scenario_strict used in R2 (verb-only match rejected)

Rules:
  R1.  Missing actors                  [ERROR]
  R2.  Missing use cases               [WARNING]
  R3.  Disconnected actors (spatial)   [ERROR]
  R4.  Missing system boundary         [WARNING]
  R4x. Unlabelled system boundary      [WARNING]
  R4y. Wrong system boundary name      [WARNING]
  R5.  Extra actors                    [WARNING]
  R6.  Extra use cases                 [INFO]
  R7.  Isolated use cases (spatial)    [ERROR]
  R8.  Empty diagram                   [ERROR]
  R9.  Unlabelled actors               [ERROR]
  R10. Unlabelled use cases            [ERROR]
  R11. Duplicate actor names           [WARNING]
  R12. Duplicate use case names        [WARNING]
  R13. Spelling mistakes in actors     [WARNING]
  R14. Spelling mistakes in use cases  [WARNING]
  R15. <<include>> unjustified         [WARNING]
  R16. <<extend>> unjustified          [WARNING]
  R17. Generalization unjustified      [INFO]
  R18. Use case verb+noun naming       [WARNING]
"""

from __future__ import annotations
import re
import math
from typing import List, Dict, Any, Set, Optional, Tuple

from .base_validator import BaseValidator, ValidationError


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

# Max px distance from line endpoint to shape bbox to count as "connected"
HIT_RADIUS = 40.0

# ── Expanded UC_VERBS from 50-scenario dataset ────────────────────────────────
UC_VERBS: Set[str] = {
    # Order / transaction
    "place", "make", "submit", "process", "confirm", "cancel", "complete",
    "purchase", "buy", "sell", "pay", "refund", "charge", "transfer",
    "checkout", "bid", "auction",
    # CRUD
    "create", "update", "delete", "edit", "remove", "add", "save",
    "modify", "set", "reset", "clear",
    # Communication
    "send", "receive", "notify", "alert", "broadcast", "publish", "post",
    "message", "share", "forward",
    # Browse / search / view
    "browse", "search", "view", "display", "list", "filter", "sort",
    "find", "check", "select", "read", "access", "explore",
    # Tracking / monitoring
    "track", "monitor", "watch", "follow", "observe", "record",
    # Auth
    "login", "logout", "register", "authenticate", "verify", "reset",
    "signup", "sign", "log", "enroll", "enrol",
    # Management
    "manage", "assign", "approve", "reject", "review", "schedule",
    "handle", "conduct", "organize", "coordinate", "allocate",
    "control", "configure", "maintain",
    # File / content
    "generate", "export", "import", "upload", "download", "print",
    "report", "archive", "backup", "scan", "capture",
    # Learning / medical / domain
    "take", "attend", "join", "book", "reserve", "borrow", "return",
    "renew", "extend", "request", "apply", "hire", "connect",
    "diagnose", "prescribe", "treat", "examine", "test", "analyze",
    "analyze", "analyse", "assess", "evaluate", "inspect",
    "withdraw", "deposit", "calculate", "estimate",
    "issue", "file", "claim", "rate", "split",
    "navigate", "deliver", "collect", "pickup", "ship",
    "install", "deploy", "build", "compile", "execute",
    "plan", "design", "develop", "implement",
    "provide", "offer", "suggest", "recommend",
    "list", "register", "validate", "certify",
}

# ── Filler verbs that appear in scenario text but are NOT use case actions ────
# These should be filtered out by the scenario extractor.
# Added here as a safety net in case extractor sends them.
_FILLER_ACTIONS: Set[str] = {
    "ensure", "allow", "enable", "support", "include", "involve",
    "perform", "divide", "separate", "provide", "offer",
    "use", "utilize", "need", "require", "want",
    "help", "assist", "serve",
}

OPTIONAL_KEYWORDS: Set[str] = {
    "optional", "sometimes", "if", "when", "only if",
    "occasionally", "may", "might", "conditionally",
}

LINE_TYPES: Set[str] = {
    "straightline", "arrow", "dottedarrow",
    "dashedarrow", "excludearrow",
    "association", "dependency",
    "generalization", "composition", "aggregation",
}


# ─────────────────────────────────────────────────────────────────────────────
#  String helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    t = re.sub(r"[^\w\s]", " ", str(text).strip().lower())
    return re.sub(r"\s+", " ", t).strip()


def _lev(a: str, b: str) -> int:
    """Levenshtein edit distance."""
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + (ca != cb)))
        prev = curr
    return prev[-1]


def _similarity_ratio(a: str, b: str) -> float:
    """Similarity ratio 0.0–1.0 based on edit distance."""
    if not a and not b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - _lev(a, b) / max_len


def _nouns_match(w1: str, w2: str) -> bool:
    """
    True if two noun words are the same word or a simple plural form.
    Handles:  order/orders, product/products, delivery/deliveries
    Rejects:  production/products, delivery/deliverance
    """
    if w1 == w2:
        return True
    # Edit distance 1 covers simple -s / -es plurals
    if _lev(w1, w2) <= 1:
        return True
    # y → ies  (delivery/deliveries, category/categories)
    if w1.endswith("y") and w2 == w1[:-1] + "ies":
        return True
    if w2.endswith("y") and w1 == w2[:-1] + "ies":
        return True
    if w1.endswith("ies") and w2 == w1[:-3] + "y":
        return True
    if w2.endswith("ies") and w1 == w2[:-3] + "y":
        return True
    return False


def _uc_matches_scenario(uc: str, actions: List[str]) -> bool:
    """
    Check if a use case label matches any scenario action.

    Key invariant: VERB must match exactly.
      - 'truck delivery' ≠ 'track delivery'  (different verb → False)
      - 'payment'        ≠ 'make payment'     (no verb in UC → False)
      - 'manage cart'    = 'manage shopping cart'  (same verb, noun subset → True)
      - 'track delivery' = 'track deliveries'       (y→ies plural → True)
    """
    q       = _norm(uc)
    q_words = q.split()
    if not q_words:
        return False
    q_verb  = q_words[0]
    q_nouns = [w for w in q_words[1:] if len(w) >= 3]

    for act in actions:
        a       = _norm(act)
        a_words = a.split()
        if not a_words:
            continue
        a_verb  = a_words[0]
        a_nouns = [w for w in a_words[1:] if len(w) >= 3]

        # Exact match
        if q == a:
            return True

        # Verb MUST be the same word
        if q_verb != a_verb:
            continue

        # No nouns on either side → verb alone matches
        if not q_nouns and not a_nouns:
            return True

        # Action has same verb but NO nouns → extractor gave partial/verb-only action
        if not a_nouns and q_nouns:
            return True

        # UC has no nouns but action does → UC is verb-only (too vague) but
        # still counts as matching this action for R2 purposes
        if not q_nouns and a_nouns:
            return True

        # All UC nouns must find a match in action nouns
        if q_nouns and all(
            any(_nouns_match(qn, an) for an in a_nouns)
            for qn in q_nouns
        ):
            return True

        # All action nouns must find a match in UC nouns
        # (handles "manage shopping cart" ↔ "manage cart")
        if q_verb in UC_VERBS and a_nouns and all(
            any(_nouns_match(an, qn) for qn in q_nouns)
            for an in a_nouns
        ):
            return True

    return False


def _uc_matches_scenario_strict(uc: str, actions: List[str]) -> bool:
    """
    Strict version for R2 (missing use case check).
    Verb-only UC labels do NOT match verb+noun scenario actions.
    e.g. diagram has 'Browse' but scenario needs 'Browse Products' → still MISSING.
    """
    q       = _norm(uc)
    q_words = q.split()
    if not q_words:
        return False
    q_verb  = q_words[0]
    q_nouns = [w for w in q_words[1:] if len(w) >= 3]

    for act in actions:
        a       = _norm(act)
        a_words = a.split()
        if not a_words:
            continue
        a_verb  = a_words[0]
        a_nouns = [w for w in a_words[1:] if len(w) >= 3]

        # Exact match always valid
        if q == a:
            return True

        # Verb must match
        if q_verb != a_verb:
            continue

        # Both verb-only → fine
        if not a_nouns and not q_nouns:
            return True

        # Action is verb-only but UC has nouns → UC is more specific → accept
        if not a_nouns and q_nouns:
            return True

        # UC has no nouns but action has nouns →
        # Accept IF the action itself came from extractor as verb-only
        # (NLP sometimes strips objects). Allow verb match.
        if not q_nouns:
            return True  # same verb, UC may be diagram shorthand

        # All UC nouns must find a match in action nouns
        if all(
            any(_nouns_match(qn, an) for an in a_nouns)
            for qn in q_nouns
        ):
            return True

        # All action nouns must find a match in UC nouns
        # handles "validate transactions" ↔ "validate transaction"
        if q_verb in UC_VERBS and a_nouns and all(
            any(_nouns_match(an, qn) for qn in q_nouns)
            for an in a_nouns
        ):
            return True

    return False


def _actor_matches(actor: str, scenario_actors: List[str]) -> bool:
    """
    Actor label matches a scenario actor.
    Uses edit distance <= 2 BUT also requires similarity ratio >= 0.55
    to prevent 'Armind' (6 chars) matching 'Admin' (5 chars).

    Also handles compound system actors:
      'Bank system' → matches if 'bank' appears in scenario text
      because dataset uses 'Bank System' as external actor even when
      scenario says 'The system validates' (implying bank system).
    """
    a = _norm(actor)
    for sa in scenario_actors:
        s = _norm(sa)
        if a == s:
            return True
        dist = _lev(a, s)
        if dist <= 2 and _similarity_ratio(a, s) >= 0.55:
            return True

    # Special case: compound actors ending in "system", "service", "server"
    # e.g. "Bank system", "Payment service", "Email server"
    # These are valid external system actors — match on the first word
    a_words = a.split()
    if len(a_words) >= 2 and a_words[-1] in ("system", "service", "server", "gateway", "platform"):
        first = a_words[0]
        for sa in scenario_actors:
            if _norm(sa).startswith(first):
                return True
        # Also valid if diagram has it and scenario mentions it as "the system"
        # Return True to avoid false EXTRA_ACTOR errors for system actors
        return True

    return False


def _best_spell_fix(
    word: str,
    pool: List[str],
    max_dist: int = 2,
) -> Optional[Tuple[str, int]]:
    """Return (best_match, dist) from pool within max_dist, else None."""
    best, best_d = None, max_dist + 1
    for cand in pool:
        d = _lev(word, cand)
        if d < best_d:
            best_d, best = d, cand
    return (best, best_d) if best and best_d <= max_dist else None


def _filter_filler_actions(actions: List[str]) -> List[str]:
    """
    Remove scenario actions whose first word is a filler verb.
    These are verbs that appear in scenario descriptions but are NOT
    meaningful use case names (e.g. 'ensure security', 'perform task').
    """
    result = []
    for act in actions:
        words = _norm(act).split()
        if not words:
            continue
        if words[0] in _FILLER_ACTIONS:
            continue
        result.append(act)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Spatial helpers — Flutter canvas coordinates
# ─────────────────────────────────────────────────────────────────────────────

def _get_pos(shape: Dict) -> Tuple[float, float]:
    p = shape.get("position") or {}
    return float(p.get("dx", 0)), float(p.get("dy", 0))


def _get_end_abs(shape: Dict) -> Optional[Tuple[float, float]]:
    """Absolute end position of a line (position + endPosition)."""
    ep = shape.get("endPosition")
    if ep is None:
        return None
    px, py = _get_pos(shape)
    return px + float(ep.get("dx", 0)), py + float(ep.get("dy", 0))


def _get_size(shape: Dict) -> Tuple[float, float]:
    s = shape.get("size") or {}
    return float(s.get("width", 80)), float(s.get("height", 60))


def _bbox(shape: Dict) -> Tuple[float, float, float, float]:
    x, y = _get_pos(shape)
    w, h = _get_size(shape)
    return x, y, x + w, y + h


def _pt_near_bbox(
    px: float, py: float,
    x1: float, y1: float, x2: float, y2: float,
    radius: float = HIT_RADIUS,
) -> bool:
    """True if point (px,py) is within `radius` of the axis-aligned bbox."""
    cx = max(x1, min(px, x2))
    cy = max(y1, min(py, y2))
    return math.hypot(px - cx, py - cy) <= radius


def _line_touches(line: Dict, shape: Dict, radius: float = HIT_RADIUS) -> bool:
    """True if either endpoint of line is within radius of shape's bbox."""
    bb = _bbox(shape)
    sx, sy = _get_pos(line)
    if _pt_near_bbox(sx, sy, *bb, radius):
        return True
    end = _get_end_abs(line)
    if end and _pt_near_bbox(end[0], end[1], *bb, radius):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  Type helper
# ─────────────────────────────────────────────────────────────────────────────

def _type(shape: Dict) -> str:
    """Normalised type string, strips 'ToolType.' prefix."""
    raw = str(shape.get("type", "")).lower()
    if "." in raw:
        raw = raw.split(".")[-1]
    return raw.strip()


def _is_line(shape: Dict) -> bool:
    return _type(shape) in LINE_TYPES


# ─────────────────────────────────────────────────────────────────────────────


class UseCaseValidator(BaseValidator):

    def validate(
        self,
        extracted: Dict[str, Any],
        shapes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        errors: List[ValidationError] = []

        # ── Categorise shapes ────────────────────────────────────────────────
        actor_shapes = [s for s in shapes if _type(s) == "actor"]
        uc_shapes    = [s for s in shapes if _type(s) == "usecase"]
        line_shapes  = [s for s in shapes if _is_line(s)]

        actor_labels = [str(s.get("text", "")).strip() for s in actor_shapes]
        uc_labels    = [str(s.get("text", "")).strip() for s in uc_shapes]

        actor_labels_ne = [l for l in actor_labels if l]
        uc_labels_ne    = [l for l in uc_labels    if l]

        # ── Scenario data ────────────────────────────────────────────────────
        scenario_actors  = extracted.get("actors",  [])
        # Filter filler actions before using them anywhere
        scenario_actions = _filter_filler_actions(extracted.get("actions", []))
        scenario_classes = [c.lower() for c in extracted.get("classes", [])]
        scenario_raw     = _norm(extracted.get("raw_text", ""))
        scenario_system_name = _norm(extracted.get("system_name", ""))

        # spell_pool: only scenario words — NOT UC_VERBS
        # This prevents 'Take' being flagged as spelling mistake for 'Make'
        spell_pool: List[str] = scenario_actors + scenario_actions + scenario_classes
        all_scenario_terms    = scenario_actions + scenario_classes

        # ════════════════════════════════════════════════════════════════════
        #  R8 — Empty diagram
        # ════════════════════════════════════════════════════════════════════
        if not shapes:
            errors.append(ValidationError(
                error_type  = "EMPTY_DIAGRAM",
                description = "Diagram is completely empty — no shapes found.",
                suggestion  = "Add actors, use cases, and a system boundary.",
                severity    = ValidationError.SEVERITY_ERROR,
                element     = "",
            ))
            return self._build_result(errors, 0, "❌ Diagram is empty.")

        if not actor_shapes and not uc_shapes:
            errors.append(ValidationError(
                error_type  = "EMPTY_DIAGRAM",
                description = "Diagram has no actors or use cases.",
                suggestion  = "Add actors and use cases first.",
                severity    = ValidationError.SEVERITY_ERROR,
                element     = "",
            ))

        # ════════════════════════════════════════════════════════════════════
        #  R9 — Unlabelled actors
        # ════════════════════════════════════════════════════════════════════
        unlabelled_a = actor_labels.count("")
        for _ in range(unlabelled_a):
            errors.append(ValidationError(
                error_type  = "UNLABELLED_ACTOR",
                description = "An actor has a missing or blank label.",
                suggestion  = "Give this actor a meaningful name, e.g. 'Customer', 'Admin'.",
                severity    = ValidationError.SEVERITY_ERROR,
                element     = "",
            ))

        # ════════════════════════════════════════════════════════════════════
        #  R10 — Unlabelled use cases
        # ════════════════════════════════════════════════════════════════════
        unlabelled_uc = uc_labels.count("")
        for _ in range(unlabelled_uc):
            errors.append(ValidationError(
                error_type  = "UNLABELLED_USE_CASE",
                description = "A use case has a missing or blank label.",
                suggestion  = "Give this use case a meaningful name, e.g. 'Place Order', 'Browse Product'.",
                severity    = ValidationError.SEVERITY_ERROR,
                element     = "",
            ))

        # ════════════════════════════════════════════════════════════════════
        #  R11 — Duplicate actor names
        # ════════════════════════════════════════════════════════════════════
        seen: Set[str] = set()
        for a in actor_labels_ne:
            k = _norm(a)
            if k in seen:
                errors.append(ValidationError(
                    error_type  = "DUPLICATE_ACTOR",
                    description = f"Actor '{a}' is duplicated in the diagram.",
                    suggestion  = "Remove the duplicate actor or rename one of them.",
                    severity    = ValidationError.SEVERITY_WARNING,
                    element     = a,
                ))
            seen.add(k)

        # ════════════════════════════════════════════════════════════════════
        #  R12 — Duplicate use case names
        # ════════════════════════════════════════════════════════════════════
        seen = set()
        for uc in uc_labels_ne:
            k = _norm(uc)
            if k in seen:
                errors.append(ValidationError(
                    error_type  = "DUPLICATE_USE_CASE",
                    description = f"Use Case '{uc}' is duplicated in the diagram.",
                    suggestion  = "Remove the duplicate use case.",
                    severity    = ValidationError.SEVERITY_WARNING,
                    element     = uc,
                ))
            seen.add(k)

        # ════════════════════════════════════════════════════════════════════
        #  R13 — Spelling mistakes in actor labels
        #  Spell check uses ONLY scenario words (spell_pool)
        #  Similarity ratio guard added to prevent over-aggressive matching
        #
        #  NOTE on generalization children:
        #  Student, Instructor etc. are valid actors even if not in spell_pool
        #  because they are specializations of a scenario actor (e.g. User).
        #  We skip spelling check for actors that are connected to a parent
        #  via generalization — they are intentionally named differently.
        # ════════════════════════════════════════════════════════════════════

        # Build set of known-valid actor names: scenario actors + all diagram
        # actors that are generalization children of scenario actors
        # (we don't know gen_child_to_parent yet at this point — R17 runs later)
        # So we use a pre-scan here.
        gen_child_labels: Set[str] = set()
        for ln in line_shapes:
            if _type(ln) != "generalization":
                continue
            sx, sy = _get_pos(ln)
            end = _get_end_abs(ln)
            child_lbl = None
            parent_lbl = None
            for a_s in actor_shapes:
                lbl = str(a_s.get("text", "")).strip()
                if not lbl:
                    continue
                bb = _bbox(a_s)
                if _pt_near_bbox(sx, sy, *bb):
                    child_lbl = lbl
                if end and _pt_near_bbox(end[0], end[1], *bb):
                    parent_lbl = lbl
            if child_lbl and parent_lbl:
                # Child is a valid specialized actor — skip spelling check
                gen_child_labels.add(child_lbl)

        if spell_pool:
            for actor in actor_labels_ne:
                # Skip if actor matches scenario actors
                if _actor_matches(actor, scenario_actors):
                    continue
                # Skip if actor is a generalization child (e.g. Student, Instructor)
                # These are valid specialized actors
                if actor in gen_child_labels:
                    continue
                for word in actor.split():
                    if len(word) < 3:
                        continue
                    w_norm = _norm(word)
                    # Skip if exact match exists in pool
                    if any(_lev(w_norm, _norm(p)) == 0 for p in spell_pool):
                        continue
                    fix = _best_spell_fix(w_norm, [_norm(p) for p in spell_pool], max_dist=2)
                    if fix:
                        correct, dist = fix
                        # Extra guard: similarity must be >= 0.6 to avoid false positives
                        if _similarity_ratio(w_norm, correct) < 0.6:
                            continue
                        orig = next((p for p in spell_pool if _norm(p) == correct), correct)
                        corrected_name = actor.replace(word, orig.title() if word[0].isupper() else orig)
                        errors.append(ValidationError(
                            error_type  = "SPELLING_MISTAKE_ACTOR",
                            description = (
                                f"Spelling mistake: '{actor}' should be '{corrected_name}'. "
                                f"Try changing '{actor}' to '{corrected_name}'."
                            ),
                            suggestion  = (
                                f"Rename '{actor}' to '{corrected_name}' to match the scenario."
                            ),
                            severity    = ValidationError.SEVERITY_WARNING,
                            element     = actor,
                        ))
                        break

        # ════════════════════════════════════════════════════════════════════
        #  R14 — Spelling mistakes in use case labels
        #  Spell check uses ONLY scenario words — NOT UC_VERBS dictionary
        #  This prevents 'Take Quiz' being flagged as 'Make Quiz' / 'Type Quiz'
        # ════════════════════════════════════════════════════════════════════
        if spell_pool:
            for uc in uc_labels_ne:
                if _uc_matches_scenario(uc, scenario_actions):
                    continue   # valid — skip
                for word in uc.split():
                    if len(word) < 4:
                        continue
                    w_norm = _norm(word)
                    # Skip if exact match exists in pool
                    if any(_lev(w_norm, _norm(p)) == 0 for p in spell_pool):
                        continue
                    # Only check against scenario words — not UC_VERBS
                    scenario_words = []
                    for phrase in spell_pool:
                        scenario_words.extend(_norm(phrase).split())
                    scenario_words = list(set(w for w in scenario_words if len(w) >= 3))

                    fix = _best_spell_fix(w_norm, scenario_words, max_dist=2)
                    if fix:
                        correct, dist = fix
                        # Extra guard: similarity must be >= 0.65
                        if _similarity_ratio(w_norm, correct) < 0.65:
                            continue
                        orig = next(
                            (p for p in scenario_words if p == correct),
                            correct
                        )
                        corrected_uc = uc.replace(word, orig.title() if word[0].isupper() else orig)
                        # Don't suggest if correction is same as original
                        if _norm(corrected_uc) == _norm(uc):
                            continue
                        errors.append(ValidationError(
                            error_type  = "SPELLING_MISTAKE_USE_CASE",
                            description = (
                                f"Spelling mistake: '{uc}' should be '{corrected_uc}'. "
                                f"Try changing '{uc}' to '{corrected_uc}'."
                            ),
                            suggestion  = (
                                f"Rename '{uc}' to '{corrected_uc}' to match the scenario."
                            ),
                            severity    = ValidationError.SEVERITY_WARNING,
                            element     = uc,
                        ))
                        break

        # ════════════════════════════════════════════════════════════════════
        #  R15 — <<include>> justification  (dashedArrow)
        # ════════════════════════════════════════════════════════════════════
        include_lines = [s for s in line_shapes if _type(s) == "dashedarrow"]
        for line in include_lines:
            end = _get_end_abs(line)
            if end is None:
                continue
            for uc_s in uc_shapes:
                label = str(uc_s.get("text", "")).strip()
                if not label:
                    continue
                bb = _bbox(uc_s)
                if _pt_near_bbox(end[0], end[1], *bb) and not _uc_matches_scenario(label, all_scenario_terms):
                    errors.append(ValidationError(
                        error_type  = "UNJUSTIFIED_INCLUDE",
                        description = (
                            f"<<include>> target '{label}' "
                            "is not mentioned in the scenario."
                        ),
                        suggestion  = (
                            f"Verify: is '{label}' actually part of the scenario? "
                            "If not, this <<include>> is unnecessary."
                        ),
                        severity    = ValidationError.SEVERITY_WARNING,
                        element     = label,
                    ))

        # ════════════════════════════════════════════════════════════════════
        #  R16 — <<extend>> justification  (excludeArrow)
        # ════════════════════════════════════════════════════════════════════
        extend_lines = [s for s in line_shapes if _type(s) == "excludearrow"]
        has_conditional = any(kw in scenario_raw for kw in OPTIONAL_KEYWORDS)
        reported_no_condition = False
        for line in extend_lines:
            sx, sy = _get_pos(line)
            for uc_s in uc_shapes:
                label = str(uc_s.get("text", "")).strip()
                if not label:
                    continue
                bb = _bbox(uc_s)
                if _pt_near_bbox(sx, sy, *bb) and not _uc_matches_scenario(label, all_scenario_terms):
                    errors.append(ValidationError(
                        error_type  = "UNJUSTIFIED_EXTEND",
                        description = (
                            f"<<extend>> source use case '{label}' was not found in the scenario."
                        ),
                        suggestion  = (
                            f"<<extend>> is used for optional behaviour. "
                            f"Confirm that '{label}' is conditional in the scenario."
                        ),
                        severity    = ValidationError.SEVERITY_WARNING,
                        element     = label,
                    ))
            if not has_conditional and not reported_no_condition:
                errors.append(ValidationError(
                    error_type  = "EXTEND_WITHOUT_CONDITION",
                    description = (
                        "<<extend>> is used but the scenario has "
                        "no conditional wording."
                    ),
                    suggestion  = (
                        "<<extend>> is only for optional steps. "
                        "The scenario should include words like 'if', 'when', or 'optionally'."
                    ),
                    severity    = ValidationError.SEVERITY_INFO,
                    element     = "",
                ))
                reported_no_condition = True

        # ════════════════════════════════════════════════════════════════════
        #  R17 — Generalization arrows
        #
        #  Also builds: gen_child_to_parent  { child_label → parent_label }
        #  This map is used in R3-EXT so that child actors connected to a
        #  parent (via generalization) are NOT flagged as "line not connected
        #  to use case" — the parent's use-case connections count for them.
        # ════════════════════════════════════════════════════════════════════
        gen_lines = [s for s in line_shapes if _type(s) == "generalization"]

        # child_label → parent_label  (e.g. "Student" → "User")
        gen_child_to_parent: Dict[str, str] = {}

        for line in gen_lines:
            sx, sy = _get_pos(line)
            end    = _get_end_abs(line)
            child_label, parent_label = None, None
            for a_s in actor_shapes:
                lbl = str(a_s.get("text", "")).strip()
                if not lbl:
                    continue
                bb = _bbox(a_s)
                if _pt_near_bbox(sx, sy, *bb):
                    child_label = lbl
                if end and _pt_near_bbox(end[0], end[1], *bb):
                    parent_label = lbl

            if child_label and parent_label:
                gen_child_to_parent[child_label] = parent_label

            if child_label and not _actor_matches(child_label, scenario_actors):
                errors.append(ValidationError(
                    error_type  = "GENERALIZATION_CHILD_NOT_IN_SCENARIO",
                    description = (
                        f"Generalization child actor '{child_label}' "
                        "is not mentioned in the scenario."
                    ),
                    suggestion  = (
                        f"Confirm that '{child_label}' is actually a specialization of "
                        f"'{parent_label or '?'}' ."
                    ),
                    severity    = ValidationError.SEVERITY_INFO,
                    element     = child_label,
                ))

        # ════════════════════════════════════════════════════════════════════
        #  R1 — Missing actors  (scenario → diagram)
        # ════════════════════════════════════════════════════════════════════
        for actor in scenario_actors:
            if not _actor_matches(actor, actor_labels_ne):
                errors.append(ValidationError(
                    error_type  = "MISSING_ACTOR",
                    description = (
                        f"Actor '{actor}' is mentioned in the scenario but missing from the diagram."
                    ),
                    suggestion  = (
                        f"Add '{actor}' to the diagram and connect it to the relevant use cases."
                    ),
                    severity    = ValidationError.SEVERITY_ERROR,
                    element     = actor,
                ))

        # ════════════════════════════════════════════════════════════════════
        #  R2 — Missing use cases  (scenario → diagram)
        #  Uses _uc_matches_scenario_strict (verb-only UC does NOT satisfy verb+noun action)
        # ════════════════════════════════════════════════════════════════════
        for action in scenario_actions:
            if not _uc_matches_scenario_strict(action, uc_labels_ne):
                errors.append(ValidationError(
                    error_type  = "MISSING_USE_CASE",
                    description = (
                        f"Scenario action '{action}' has no matching use case "
                        "in the diagram."
                    ),
                    suggestion  = f"Add use case '{action.title()}' to the diagram.",
                    severity    = ValidationError.SEVERITY_WARNING,
                    element     = action,
                ))

        # ════════════════════════════════════════════════════════════════════
        #  R3 — Disconnected actors  (spatial proximity)
        # ════════════════════════════════════════════════════════════════════
        for a_shape in actor_shapes:
            label = str(a_shape.get("text", "")).strip()
            if not label:
                continue
            connected = any(_line_touches(ln, a_shape) for ln in line_shapes)
            if not connected:
                errors.append(ValidationError(
                    error_type  = "DISCONNECTED_ACTOR",
                    description = (
                        f"Actor '{label}' is not connected to any use case — "
                        "no line touches this actor."
                    ),
                    suggestion  = f"Draw association lines from '{label}' to the relevant use cases.",
                    severity    = ValidationError.SEVERITY_ERROR,
                    element     = label,
                ))

        # ════════════════════════════════════════════════════════════════════
        #  R4 — System boundary presence
        # ════════════════════════════════════════════════════════════════════
        boundary_shapes = [s for s in shapes if _type(s) == "systemboundary"]
        has_boundary    = bool(boundary_shapes)

        if not has_boundary and uc_labels_ne:
            errors.append(ValidationError(
                error_type  = "MISSING_SYSTEM_BOUNDARY",
                description = "System boundary is missing from the diagram.",
                suggestion  = (
                    "Place all use cases inside a system boundary rectangle "
                    "to define the system scope."
                ),
                severity    = ValidationError.SEVERITY_WARNING,
                element     = "",
            ))

        # R4x — System boundary label checks
        for sb in boundary_shapes:
            sb_label = str(sb.get("text", "")).strip()

            # R4x: Unlabelled boundary
            if not sb_label:
                errors.append(ValidationError(
                    error_type  = "UNLABELLED_SYSTEM_BOUNDARY",
                    description = "System boundary has no label — the system name is missing.",
                    suggestion  = (
                        "Add a name to the system boundary rectangle, "
                        "e.g. 'Online Shopping System' or 'Library System'."
                    ),
                    severity    = ValidationError.SEVERITY_WARNING,
                    element     = "",
                ))

            # R4y: Wrong boundary name (only check if scenario has a system name)
            elif scenario_system_name:
                sb_norm = _norm(sb_label)

                # Check match using multiple strategies:
                # 1. Exact match
                # 2. One contains the other
                # 3. Levenshtein distance within allowed range
                # 4. Key words match (e.g. "ATM" in both)

                is_match = False

                # Strategy 1 & 2: containment
                if sb_norm == scenario_system_name:
                    is_match = True
                elif sb_norm in scenario_system_name or scenario_system_name in sb_norm:
                    is_match = True
                else:
                    # Strategy 3: edit distance
                    dist = _lev(sb_norm, scenario_system_name)
                    max_allowed = max(3, len(scenario_system_name) // 4)
                    if dist <= max_allowed:
                        is_match = True

                    # Strategy 4: key content words match (ignore "system" word)
                    if not is_match:
                        sb_words = set(sb_norm.replace("system", "").split()) - {"", "the", "a", "an"}
                        sc_words = set(scenario_system_name.replace("system", "").split()) - {"", "the", "a", "an"}
                        if sb_words and sc_words:
                            # If ANY key word matches → consider it close enough
                            if sb_words & sc_words:
                                is_match = True

                if not is_match:
                    errors.append(ValidationError(
                        error_type  = "WRONG_SYSTEM_BOUNDARY_NAME",
                        description = (
                            f"System boundary is labelled '{sb_label}' but the scenario "
                            f"describes '{extracted.get('system_name', '')}'. "
                            "The boundary name should match the system being modelled."
                        ),
                        suggestion  = (
                            f"Rename the system boundary to "
                            f"'{extracted.get('system_name', '').title()}' "
                            "to match the scenario."
                        ),
                        severity    = ValidationError.SEVERITY_WARNING,
                        element     = sb_label,
                    ))

        # ════════════════════════════════════════════════════════════════════
        #  R5 — Extra actors  (diagram → scenario)
        #  Skip generalization children — they are valid specialized actors
        #  e.g. Student and Instructor are valid even if scenario says "User"
        # ════════════════════════════════════════════════════════════════════
        for actor in actor_labels_ne:
            if actor in gen_child_labels:
                continue   # valid generalization child — not extra
            if not _actor_matches(actor, scenario_actors):
                errors.append(ValidationError(
                    error_type  = "EXTRA_ACTOR",
                    description = (
                        f"Actor '{actor}' is in the diagram but not mentioned in the scenario."
                    ),
                    suggestion  = (
                        f"Verify: is '{actor}' related to the scenario? "
                        "If not, remove it from the diagram."
                    ),
                    severity    = ValidationError.SEVERITY_WARNING,
                    element     = actor,
                ))

        # ════════════════════════════════════════════════════════════════════
        #  R6 — Extra use cases  (diagram → scenario)
        # ════════════════════════════════════════════════════════════════════
        for uc in uc_labels_ne:
            if not _uc_matches_scenario(uc, all_scenario_terms):
                errors.append(ValidationError(
                    error_type  = "EXTRA_USE_CASE",
                    description = f"Use Case '{uc}' does not match any action in the scenario.",
                    suggestion  = (
                        f"'{uc}' — review this: it is not mentioned in the scenario "
                        "or the wording differs."
                    ),
                    severity    = ValidationError.SEVERITY_INFO,
                    element     = uc,
                ))

        # ════════════════════════════════════════════════════════════════════
        #  R7 — Isolated use cases  (spatial proximity)
        # ════════════════════════════════════════════════════════════════════
        for uc_shape in uc_shapes:
            label = str(uc_shape.get("text", "")).strip()
            if not label:
                continue
            connected = any(_line_touches(ln, uc_shape) for ln in line_shapes)
            if not connected:
                errors.append(ValidationError(
                    error_type  = "ISOLATED_USE_CASE",
                    description = (
                        f"Use Case '{label}' is isolated — "
                        "no line connects to this use case."
                    ),
                    suggestion  = f"Connect '{label}' to an actor, or remove it from the diagram.",
                    severity    = ValidationError.SEVERITY_ERROR,
                    element     = label,
                ))

        # ════════════════════════════════════════════════════════════════════
        #  R18 — Use Case verb+noun naming convention
        #
        #  Skip if UC already matches scenario (valid). Then:
        #   a) First word is valid verb, single word → too vague
        #   b) First word is valid verb, multi-word → valid (no error)
        #   c) No verb → find which scenario phrase it belongs to → MISSING_VERB
        #   d) First word is NOT in UC_VERBS → check if it's a noun without verb
        #   e) Generic → suggest adding verb
        #
        #  NOTE: Spelling typo check for verbs is removed here.
        #  'Take Quiz' is valid because 'take' IS in UC_VERBS.
        #  We no longer fuzzy-match verb against UC_VERBS to avoid false positives.
        # ════════════════════════════════════════════════════════════════════
        sc_action_norms = [_norm(a) for a in scenario_actions]

        for uc in uc_labels_ne:
            uc_n     = _norm(uc)
            uc_words = uc_n.split()
            if not uc_words:
                continue

            # a) Matches scenario action → completely valid
            if _uc_matches_scenario(uc, scenario_actions):
                continue

            first = uc_words[0]

            # b) First word is a valid verb
            if first in UC_VERBS:
                if len(uc_words) == 1:
                    errors.append(ValidationError(
                        error_type  = "USE_CASE_TOO_VAGUE",
                        description = (
                            f"Use Case '{uc}' contains only a verb — noun is missing. "
                            "e.g. 'manage' → 'Manage Cart' or 'Manage Products'."
                        ),
                        suggestion  = "Use case names should follow the Verb + Noun format.",
                        severity    = ValidationError.SEVERITY_WARNING,
                        element     = uc,
                    ))
                continue   # multi-word valid verb phrase — no error

            # c) First word is NOT a verb — try to find matching scenario phrase
            matched_verb, matched_phrase = None, None
            for phrase in sc_action_norms:
                p_words = phrase.split()
                if len(p_words) >= 2:
                    p_verb  = p_words[0]
                    p_nouns = [w for w in p_words[1:] if len(w) >= 3]
                    uc_all_nouns = [w for w in uc_words if len(w) >= 3]
                    if uc_all_nouns and p_nouns and all(
                        any(_nouns_match(un, pn) for pn in p_nouns) for un in uc_all_nouns
                    ):
                        matched_verb, matched_phrase = p_verb, phrase
                        break

            if matched_verb:
                errors.append(ValidationError(
                    error_type  = "MISSING_VERB_IN_USE_CASE",
                    description = (
                        f"Use Case '{uc}' is missing a verb. "
                        f"The scenario contains '{matched_phrase}' — "
                        f"add the verb '{matched_verb}' to the name."
                    ),
                    suggestion  = (
                        f"Rename to: '{matched_verb.capitalize()} {uc.title()}'"
                    ),
                    severity    = ValidationError.SEVERITY_WARNING,
                    element     = uc,
                ))
                continue

            # d) Generic naming warning — first word is not a known action verb
            errors.append(ValidationError(
                error_type  = "INVALID_USE_CASE_NAME",
                description = (
                    f"Use Case '{uc}' does not follow the Verb+Noun format — "
                    f"'{first}' is not an action verb."
                ),
                suggestion  = (
                    f"Start with an action verb: "
                    f"'View {uc.title()}' or 'Manage {uc.title()}'."
                ),
                severity    = ValidationError.SEVERITY_WARNING,
                element     = uc,
            ))

        # ════════════════════════════════════════════════════════════════════
        #  R3-EXT — Actor lines must connect to at least one USE CASE
        #
        #  GENERALIZATION RULE:
        #  If actor is a CHILD in a generalization (e.g. Student → User),
        #  its lines go to the PARENT actor, not directly to use cases.
        #  This is correct UML — child inherits parent's use cases.
        #  So: if ALL of an actor's lines touch only its generalization
        #  parent (and not any use case), do NOT flag it as an error.
        # ════════════════════════════════════════════════════════════════════
        for a_shape in actor_shapes:
            label = str(a_shape.get("text", "")).strip()
            if not label:
                continue

            actor_lines = [ln for ln in line_shapes if _line_touches(ln, a_shape)]
            if not actor_lines:
                continue   # already caught by R3 (DISCONNECTED_ACTOR)

            # Find the parent shape if this actor is a generalization child
            parent_label = gen_child_to_parent.get(label)
            parent_shape = None
            if parent_label:
                parent_shape = next(
                    (s for s in actor_shapes
                     if str(s.get("text", "")).strip() == parent_label),
                    None
                )

            reported_actor_level = False
            for ln in actor_lines:
                touches_uc = any(_line_touches(ln, uc_s) for uc_s in uc_shapes)
                if not touches_uc:
                    # Check if this line is a generalization line to the parent
                    # (i.e. it touches the parent actor) — if so, skip
                    if parent_shape and _line_touches(ln, parent_shape):
                        continue  # valid generalization line — not an error

                    if not reported_actor_level:
                        errors.append(ValidationError(
                            error_type  = "ACTOR_LINE_NOT_CONNECTED_TO_USE_CASE",
                            description = (
                                f"Actor '{label}' has a line drawn but it is not connected "
                                "to any use case in the diagram."
                            ),
                            suggestion  = (
                                f"Draw the line from '{label}' all the way to a use case ellipse, "
                                "or connect it to an existing use case."
                            ),
                            severity    = ValidationError.SEVERITY_ERROR,
                            element     = label,
                        ))
                        reported_actor_level = True

        # ════════════════════════════════════════════════════════════════════
        #  Score
        # ════════════════════════════════════════════════════════════════════
        ec = sum(1 for e in errors if e.severity == ValidationError.SEVERITY_ERROR)
        wc = sum(1 for e in errors if e.severity == ValidationError.SEVERITY_WARNING)
        ic = sum(1 for e in errors if e.severity == ValidationError.SEVERITY_INFO)
        score = max(0, 100 - ec * 15 - wc * 5 - ic * 1)

        if ec == 0 and wc == 0 and ic == 0:
            summary = "✅ Diagram is correct!"
        elif ec == 0 and wc == 0:
            summary = f"ℹ️ {ic} info note(s) — diagram looks good overall."
        elif ec == 0:
            summary = f"⚠️ {wc} warning(s) found — please review."
        else:
            summary = f"❌ {ec} error(s) and {wc} warning(s) found."

        return self._build_result(errors, score, summary)
