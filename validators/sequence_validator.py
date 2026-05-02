"""
validators/sequence_validator.py
─────────────────────────────────
Sequence Diagram — rule-based + offline semantic validation.

Rules:
  R1.  Scenario ke har participant diagram mein lifeline hona chahiye.
  R2.  [SEMANTIC] Scenario ke har interaction diagram mein semantically hona chahiye.
  R3.  Message arrows valid lifelines ke beech hone chahiye.
  R4.  Koi isolated lifeline nahi honi chahiye.
  R5.  Return/response messages hone chahiye (best-practice).
  R6.  Diagram mein kam az kam 2 participants hone chahiye.
  R7.  Har lifeline/object/actor ka label hona chahiye (empty box bhi error hai).
  R8.  Har message arrow ka label hona chahiye.
  R8b. Har arrow ke from aur to endpoints valid hone chahiye (None→None nahi).
  R9.  Jo lifelines messages receive karti hain unke paas activation box honi chahiye.
  R10. Diagram participants exactly scenario participants se match karein.
  R11. [NEW] Har lifeline ke end par deletion symbol (X) hona chahiye.

Key fix:
  - DiagramParser use karta hai jo 20+ different field name variants handle karta hai.
    Isi wajah se pehle label="", from="None", to="None" aa raha tha.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from .base_validator import BaseValidator, ValidationError
from .diagram_parser import DiagramParser

# ── NLP noise tokens ──────────────────────────────────────────────────────────
_NON_PARTICIPANT_TOKENS: Set[str] = {
    "button", "email", "password", "field", "form", "input", "checkbox",
    "dropdown", "link", "icon", "screen", "page", "modal", "popup", "toast",
    "detail", "details", "record", "records", "response", "confirmation",
    "message", "notification", "alert", "result", "results", "data",
    "request", "reply", "error", "status", "token", "session", "cookie",
    "login", "logout", "signup", "search", "click", "submit", "send",
    "fetch", "load", "open", "close", "update", "delete", "create",
    "dashboard",
}


class SequenceDiagramValidator(BaseValidator):

    def __init__(self, semantic_threshold: float = 0.45) -> None:
        super().__init__()
        self._threshold        = semantic_threshold
        self._semantic_checker = None   # lazy load on first use
        self._parser           = DiagramParser()

    # ═══════════════════════════════════════════════════════════════════════
    # Public entry point
    # ═══════════════════════════════════════════════════════════════════════

    def validate(
        self,
        extracted: Dict[str, Any],
        shapes:    List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        errors: List[ValidationError] = []

        # ── STEP 1: Normalize raw shapes from drawing app ─────────────────────
        # This fixes: label="", from="None", to="None", id=null, etc.
        shapes = self._parser.normalize(shapes)

        # ── STEP 2: Categorize shapes ─────────────────────────────────────────
        lifeline_shapes    = self._get_lifelines(shapes)
        actor_shapes       = self._get_actors(shapes)       # ← actors alag bhi read karo
        message_shapes     = self._get_messages(shapes)
        activation_shapes  = self._get_activations(shapes)
        deletion_shapes    = self._get_deletions(shapes)
        fragment_shapes    = self._get_fragments(shapes)    # ← alt/opt/loop/par boxes
        guard_shapes       = self._get_guards(shapes)       # ← [condition] operands

        # Combine lifeline + actor labels together
        lifeline_labels: List[str] = self._valid_labels(lifeline_shapes)
        actor_labels:    List[str] = self._valid_labels(actor_shapes)
        # Merge both so validation uses all participant labels
        all_participant_labels: List[str] = list(set(lifeline_labels + actor_labels))

        all_labels:      List[str] = self._valid_labels(shapes)

        diagram_message_labels: List[str] = [
            s["label"] for s in message_shapes if s.get("label")
        ]

        # ── STEP 3: Parse scenario ────────────────────────────────────────────
        raw_actors   = extracted.get("actors",       [])
        raw_objects  = extracted.get("objects",      [])
        interactions = extracted.get("interactions", [])

        expected_lifelines: List[str] = list({
            p for p in
            self._filter_participants(raw_actors) +
            self._filter_participants(raw_objects)
            if p
        })

        # ════════════════════════════════════════════════════════════════════
        # R1 — Missing lifelines (check against actors + lifelines both)
        # ════════════════════════════════════════════════════════════════════
        for participant in expected_lifelines:
            if not self._fuzzy_match(participant, all_participant_labels + all_labels):
                errors.append(ValidationError(
                    error_type  = "MISSING_LIFELINE",
                    description = (
                        f"Object/Actor '{participant}' scenario mein hai lekin "
                        f"diagram mein koi lifeline nahi."
                    ),
                    suggestion  = f"'{participant}' ke liye ek lifeline add karo.",
                    severity    = ValidationError.SEVERITY_ERROR,
                    element     = participant,
                ))

        # ════════════════════════════════════════════════════════════════════
        # R2 — Semantic interaction check
        # ════════════════════════════════════════════════════════════════════
        if interactions:
            errors.extend(
                self._check_interactions_semantic(interactions, diagram_message_labels)
            )

        # ════════════════════════════════════════════════════════════════════
        # R3 — Message arrows between valid lifelines
        # ════════════════════════════════════════════════════════════════════
        valid_pool = all_participant_labels + all_labels
        for msg in message_shapes:
            frm = msg.get("from", "")
            to  = msg.get("to",   "")

            if frm and not self._fuzzy_match(frm, valid_pool):
                errors.append(ValidationError(
                    error_type  = "INVALID_MESSAGE_SOURCE",
                    description = f"Message source '{frm}' koi valid lifeline nahi hai.",
                    suggestion  = f"'{frm}' ko diagram mein lifeline ke tor par add karo.",
                    severity    = ValidationError.SEVERITY_ERROR,
                    element     = frm,
                ))
            if to and not self._fuzzy_match(to, valid_pool):
                errors.append(ValidationError(
                    error_type  = "INVALID_MESSAGE_TARGET",
                    description = f"Message target '{to}' koi valid lifeline nahi hai.",
                    suggestion  = f"'{to}' ko diagram mein lifeline ke tor par add karo.",
                    severity    = ValidationError.SEVERITY_ERROR,
                    element     = to,
                ))

        # ════════════════════════════════════════════════════════════════════
        # R4 — Isolated lifelines
        # ════════════════════════════════════════════════════════════════════
        connected: Set[str] = set()
        for msg in message_shapes:
            if msg.get("from"): connected.add(msg["from"].lower())
            if msg.get("to"):   connected.add(msg["to"].lower())

        for ll in all_participant_labels:
            if ll.lower() not in connected:
                errors.append(ValidationError(
                    error_type  = "ISOLATED_LIFELINE",
                    description = f"Lifeline '{ll}' kisi bhi message mein participate nahi kar rahi.",
                    suggestion  = f"'{ll}' se koi message bhejo ya receive karo.",
                    severity    = ValidationError.SEVERITY_WARNING,
                    element     = ll,
                ))

        # ════════════════════════════════════════════════════════════════════
        # R5 — Return messages
        # ════════════════════════════════════════════════════════════════════
        return_msgs = [
            m for m in message_shapes
            if any(k in m.get("type", "") for k in ("return", "dashed", "dotted"))
        ]
        if message_shapes and not return_msgs:
            errors.append(ValidationError(
                error_type  = "NO_RETURN_MESSAGES",
                description = "Diagram mein koi return/response message nahi hai.",
                suggestion  = "Har request ke baad ek response (dashed arrow) draw karo.",
                severity    = ValidationError.SEVERITY_INFO,
                element     = "",
            ))

        # ════════════════════════════════════════════════════════════════════
        # R6 — No participants at all
        # ════════════════════════════════════════════════════════════════════
        if not lifeline_shapes and not actor_shapes:
            errors.append(ValidationError(
                error_type  = "NO_PARTICIPANTS",
                description = "Diagram mein koi bhi lifeline / participant nahi hai.",
                suggestion  = "Har scenario participant ke liye ek lifeline draw karo.",
                severity    = ValidationError.SEVERITY_ERROR,
                element     = "",
            ))

        # ════════════════════════════════════════════════════════════════════
        # R7 — Unlabelled lifelines  (empty box bhi error hai)
        # ════════════════════════════════════════════════════════════════════
        # DiagramParser ke baad bhi agar label empty hai — shape empty drawn hai
        for shape in lifeline_shapes + actor_shapes:
            if not shape.get("label"):
                shape_id   = shape.get("id", "unknown")
                raw_type   = shape.get("_raw_type", shape.get("type", ""))
                errors.append(ValidationError(
                    error_type  = "UNLABELLED_LIFELINE",
                    description = (
                        f"'{raw_type}' shape (id='{shape_id}') ka koi naam nahi hai. "
                        f"Empty box/actor draw kiya gaya hai."
                    ),
                    suggestion  = (
                        f"Is shape par double-tap karo aur participant ka naam likho "
                        f"(masalan 'User', 'App', 'Database')."
                    ),
                    severity    = ValidationError.SEVERITY_ERROR,
                    element     = str(shape_id),
                ))

        # ════════════════════════════════════════════════════════════════════
        # R8 — Unlabelled arrows
        # ════════════════════════════════════════════════════════════════════
        for msg in message_shapes:
            if not msg.get("label"):
                msg_id = msg.get("id", "unknown")
                frm    = msg.get("from", "") or "?"
                to     = msg.get("to",   "") or "?"
                errors.append(ValidationError(
                    error_type  = "UNLABELLED_ARROW",
                    description = (
                        f"Arrow '{frm} → {to}' (id='{msg_id}') ka koi label nahi."
                    ),
                    suggestion  = (
                        "Arrow par double-tap karo aur operation ka naam likho "
                        "jaise 'login()', 'findUser()', 'loginSuccessful'."
                    ),
                    severity    = ValidationError.SEVERITY_ERROR,
                    element     = str(msg_id),
                ))

        # ════════════════════════════════════════════════════════════════════
        # R8b — Arrow endpoints missing (from/to both empty = floating arrow)
        # ════════════════════════════════════════════════════════════════════
        for msg in message_shapes:
            frm = msg.get("from", "")
            to  = msg.get("to",   "")
            if not frm and not to:
                msg_id    = msg.get("id", "unknown")
                msg_label = msg.get("label", "") or "unlabelled"
                errors.append(ValidationError(
                    error_type  = "FLOATING_ARROW",
                    description = (
                        f"Arrow '{msg_label}' (id='{msg_id}') kisi bhi lifeline se "
                        f"connected nahi hai — dono endpoints missing hain."
                    ),
                    suggestion  = (
                        "Arrow ko source lifeline se target lifeline tak properly "
                        "connect karo. Arrow ko drag karke lifeline se attach karo."
                    ),
                    severity    = ValidationError.SEVERITY_ERROR,
                    element     = str(msg_id),
                ))
            elif not frm and to:
                msg_id = msg.get("id", "unknown")
                errors.append(ValidationError(
                    error_type  = "ARROW_MISSING_SOURCE",
                    description = (
                        f"Arrow '{msg.get('label', 'unlabelled')}' ka source "
                        f"(from) missing hai — sirf target '{to}' connected hai."
                    ),
                    suggestion  = "Arrow ka starting point kisi lifeline se connect karo.",
                    severity    = ValidationError.SEVERITY_WARNING,
                    element     = str(msg_id),
                ))
            elif frm and not to:
                msg_id = msg.get("id", "unknown")
                errors.append(ValidationError(
                    error_type  = "ARROW_MISSING_TARGET",
                    description = (
                        f"Arrow '{msg.get('label', 'unlabelled')}' ka target "
                        f"(to) missing hai — sirf source '{frm}' connected hai."
                    ),
                    suggestion  = "Arrow ka ending point kisi lifeline se connect karo.",
                    severity    = ValidationError.SEVERITY_WARNING,
                    element     = str(msg_id),
                ))

        # ════════════════════════════════════════════════════════════════════
        # R9 — Missing activation boxes
        # ════════════════════════════════════════════════════════════════════
        activated: Set[str] = set()

        for s in activation_shapes:
            # Try 1: explicit lifeline reference field (set by Flutter resolveArrowConnections)
            ref = (
                s.get("lifelineRef") or
                s.get("lifeline")    or
                s.get("on")          or
                s.get("label")       or
                ""
            )
            ref = str(ref).strip().lower()
            if ref and ref not in ("none", "null", ""):
                activated.add(ref)

        # Try 2: if no refs found but activation shapes exist,
        # match by X-position proximity to participant heads
        if activation_shapes and not activated:
            act_xs = []
            for s in activation_shapes:
                pos = s.get("position") or {}
                x = pos.get("dx") if isinstance(pos, dict) else None
                if x is not None:
                    act_xs.append(float(x))

            head_positions = []
            for s in lifeline_shapes + actor_shapes:
                pos = s.get("position") or {}
                x = pos.get("dx") if isinstance(pos, dict) else None
                lbl = s.get("label", "")
                if x is not None and lbl:
                    head_positions.append((float(x), lbl.lower()))

            if act_xs and head_positions:
                for ax in act_xs:
                    # find closest head by X
                    closest = min(head_positions, key=lambda h: abs(h[0] - ax))
                    if abs(closest[0] - ax) < 80:   # within 80px tolerance
                        activated.add(closest[1])

        # Try 3: if STILL no refs (simple app, no position) but activation
        # count >= receiving lifelines count → assume all covered
        receiving: Set[str] = {
            msg["to"].lower()
            for msg in message_shapes
            if msg.get("to")
        }

        if activation_shapes and not activated:
            if len(activation_shapes) >= len(receiving):
                # Enough activation boxes drawn — skip R9
                activated = receiving.copy()

        for ll in all_participant_labels:
            if ll.lower() in receiving and ll.lower() not in activated:
                errors.append(ValidationError(
                    error_type  = "MISSING_ACTIVATION_BOX",
                    description = (
                        f"Lifeline '{ll}' messages receive kar rahi hai lekin "
                        f"activation box nahi hai."
                    ),
                    suggestion  = (
                        f"'{ll}' par activation box (thin rectangle on lifeline) "
                        f"draw karo jab woh request process kar rahi ho."
                    ),
                    severity    = ValidationError.SEVERITY_WARNING,
                    element     = ll,
                ))

        # ════════════════════════════════════════════════════════════════════
        # R10 — Participant set match (diagram ↔ scenario)
        # ════════════════════════════════════════════════════════════════════
        diagram_set:  Set[str] = {ll.lower() for ll in all_participant_labels if ll}
        scenario_set: Set[str] = {p.lower()  for p in expected_lifelines if p}

        for extra in sorted(diagram_set - scenario_set):
            if self._fuzzy_match(extra, list(scenario_set)):
                continue
            errors.append(ValidationError(
                error_type  = "EXTRA_PARTICIPANT",
                description = (
                    f"Lifeline '{extra}' diagram mein hai lekin scenario mein "
                    f"is participant ka zikr nahi."
                ),
                suggestion  = (
                    f"'{extra}' hata do agar scenario ka hissa nahi, "
                    f"ya scenario update karo."
                ),
                severity    = ValidationError.SEVERITY_WARNING,
                element     = extra,
            ))

        for missing in sorted(scenario_set - diagram_set):
            if self._fuzzy_match(missing, list(diagram_set)):
                continue
            already_reported = any(
                e.error_type == "MISSING_LIFELINE"
                and e.element.lower() == missing.lower()
                for e in errors
            )
            if not already_reported:
                errors.append(ValidationError(
                    error_type  = "MISMATCHED_PARTICIPANT",
                    description = (
                        f"Scenario participant '{missing}' diagram mein represent nahi ho raha."
                    ),
                    suggestion  = f"'{missing}' ke liye ek labelled lifeline add karo.",
                    severity    = ValidationError.SEVERITY_ERROR,
                    element     = missing,
                ))

        # ════════════════════════════════════════════════════════════════════
        # R11 — Missing deletion symbols (X marks at lifeline ends)
        # ════════════════════════════════════════════════════════════════════
        # Sequence diagram mein har lifeline ke end par X (destroy) symbol
        # hona chahiye — yeh dikhata hai ke object destroy ho gaya.
        #
        # Check: har lifeline ke liye ek deletion shape honi chahiye.
        # Deletion shapes ka "lifeline" ya "on" field lifeline label se match kare.
        deletion_targets: Set[str] = set()
        for d in deletion_shapes:
            # Try to find which lifeline this deletion belongs to
            ref = (
                d.get("lifeline") or
                d.get("on")       or
                d.get("label")    or
                d.get("from")     or
                ""
            )
            ref = str(ref).strip().lower()
            if ref and ref not in ("none", "null"):
                deletion_targets.add(ref)

        # Use unique participant count (not sum of two lists which double-counts actors)
        _total_participants = len(all_participant_labels)
        if deletion_shapes and not deletion_targets:
            # Fallback: assume deletions are present if count >= participants count
            if len(deletion_shapes) >= _total_participants:
                # Enough deletion symbols — no error
                pass
            else:
                missing_count = _total_participants - len(deletion_shapes)
                errors.append(ValidationError(
                    error_type  = "MISSING_DELETION_SYMBOLS",
                    description = (
                        f"Diagram mein {_total_participants} lifelines hain lekin "
                        f"sirf {len(deletion_shapes)} deletion symbol(s) (X) hain. "
                        f"{missing_count} deletion symbol(s) missing hain."
                    ),
                    suggestion  = (
                        "Har lifeline ke end par ek X (destroy/deletion) symbol draw karo. "
                        "Yeh dikhata hai ke object ka kaam khatam ho gaya."
                    ),
                    severity    = ValidationError.SEVERITY_WARNING,
                    element     = "",
                ))
        elif not deletion_shapes and lifeline_shapes:
            # No deletion symbols at all
            errors.append(ValidationError(
                error_type  = "NO_DELETION_SYMBOLS",
                description = (
                    f"Diagram mein koi bhi deletion symbol (X mark) nahi hai. "
                    f"Har lifeline ke end par X hona chahiye."
                ),
                suggestion  = (
                    f"Tamam {len(lifeline_shapes)} lifelines ke end par "
                    f"deletion symbol (X / destroy marker) draw karo."
                ),
                severity    = ValidationError.SEVERITY_WARNING,
                element     = "",
            ))
        else:
            # We have references — check each lifeline
            for ll in lifeline_labels:
                if ll.lower() not in deletion_targets:
                    errors.append(ValidationError(
                        error_type  = "MISSING_DELETION_SYMBOL",
                        description = (
                            f"Lifeline '{ll}' ke end par deletion symbol (X) nahi hai."
                        ),
                        suggestion  = (
                            f"'{ll}' lifeline ke neeche ek X (destroy) symbol draw karo."
                        ),
                        severity    = ValidationError.SEVERITY_WARNING,
                        element     = ll,
                    ))

        # ════════════════════════════════════════════════════════════════════
        # R12 — Combined Fragment (alt/opt/loop/par) validation
        # ════════════════════════════════════════════════════════════════════
        # R12a: alt fragment mein kam az kam 2 operands hone chahiye
        # R12b: har fragment mein kam az kam ek message hona chahiye
        # R12c: fragment ka label valid keyword hona chahiye
        _VALID_FRAGMENT_LABELS = {"alt", "opt", "loop", "par", "break",
                                  "critical", "neg", "ref"}
        _MULTI_OPERAND_TYPES   = {"alt", "par"}  # inhe 2+ guards chahiye

        for frag in fragment_shapes:
            frag_id    = frag.get("id", "unknown")
            frag_label = str(frag.get("label", "")).strip().lower()

            # R12c — fragment label valid hona chahiye
            if frag_label and frag_label not in _VALID_FRAGMENT_LABELS:
                errors.append(ValidationError(
                    error_type  = "INVALID_FRAGMENT_TYPE",
                    description = (
                        f"Fragment (id='{frag_id}') ka label '{frag_label}' valid "
                        f"nahi hai. Valid types: alt, opt, loop, par, break, ref."
                    ),
                    suggestion  = (
                        f"Fragment ka label 'alt', 'opt', 'loop', ya 'par' hona chahiye."
                    ),
                    severity    = ValidationError.SEVERITY_ERROR,
                    element     = str(frag_id),
                ))

            # R12a — alt/par fragment mein kam az kam 2 guard operands hone chahiye
            if frag_label in _MULTI_OPERAND_TYPES:
                # Count guards that belong to this fragment (by fragmentId or proximity)
                frag_guards = [
                    g for g in guard_shapes
                    if (str(g.get("fragmentId", "")) == str(frag_id) or
                        str(g.get("fragment",   "")) == str(frag_id))
                ]
                # Fallback: if no fragmentId links, count total guards
                guards_to_check = frag_guards if frag_guards else guard_shapes

                if len(guards_to_check) < 2:
                    errors.append(ValidationError(
                        error_type  = "ALT_MISSING_OPERANDS",
                        description = (
                            f"'{frag_label.upper()}' fragment (id='{frag_id}') mein "
                            f"kam az kam 2 guard operands hone chahiye — "
                            f"jaise [condition] aur [else]."
                        ),
                        suggestion  = (
                            f"Fragment ke andar 2 sections banao: pehla "
                            f"'[condition]' ke saath aur doosra '[else]' ke saath."
                        ),
                        severity    = ValidationError.SEVERITY_ERROR,
                        element     = str(frag_id),
                    ))

            # R12b — fragment ke andar kam az kam ek message hona chahiye
            frag_messages = [
                m for m in message_shapes
                if (str(m.get("fragmentId", "")) == str(frag_id) or
                    str(m.get("fragment",   "")) == str(frag_id))
            ]
            # Fallback: if no fragmentId links, skip this check (can't determine)
            if frag_messages is not None and len(frag_messages) == 0:
                # Only report if fragmentId links exist but messages are empty
                frag_linked = any(
                    str(m.get("fragmentId", "")) == str(frag_id)
                    for m in message_shapes
                )
                if frag_linked:
                    errors.append(ValidationError(
                        error_type  = "EMPTY_FRAGMENT",
                        description = (
                            f"Fragment '{frag_label.upper()}' (id='{frag_id}') ke "
                            f"andar koi message arrow nahi hai."
                        ),
                        suggestion  = (
                            f"Fragment ke har operand mein kam az kam ek message "
                            f"arrow draw karo."
                        ),
                        severity    = ValidationError.SEVERITY_WARNING,
                        element     = str(frag_id),
                    ))

        # ════════════════════════════════════════════════════════════════════
        # R13 — Guard condition validation
        # ════════════════════════════════════════════════════════════════════
        # R13a: guard label [ ] brackets mein hona chahiye
        # R13b: guard label empty nahi hona chahiye
        # R13c: alt fragment mein [else] ya dusra guard hona chahiye

        for guard in guard_shapes:
            guard_id  = guard.get("id", "unknown")
            guard_lbl = str(guard.get("label", "")).strip()

            # R13b — guard label empty nahi hona chahiye
            if not guard_lbl or guard_lbl in ("[]", "[ ]"):
                errors.append(ValidationError(
                    error_type  = "EMPTY_GUARD",
                    description = (
                        f"Guard condition (id='{guard_id}') ka label empty hai. "
                        f"Guard mein condition likhni chahiye."
                    ),
                    suggestion  = (
                        "Guard mein condition likho jaise [x > 0], [isValid], "
                        "[else]. Brackets ke andar condition zaroor honi chahiye."
                    ),
                    severity    = ValidationError.SEVERITY_ERROR,
                    element     = str(guard_id),
                ))
                continue

            # R13a — guard label [ ] se wrapped hona chahiye
            if not (guard_lbl.startswith("[") and guard_lbl.endswith("]")):
                errors.append(ValidationError(
                    error_type  = "GUARD_MISSING_BRACKETS",
                    description = (
                        f"Guard condition '{guard_lbl}' (id='{guard_id}') square "
                        f"brackets mein nahi hai."
                    ),
                    suggestion  = (
                        f"Guard condition ko brackets mein wrap karo: '[{guard_lbl}]'."
                    ),
                    severity    = ValidationError.SEVERITY_WARNING,
                    element     = str(guard_id),
                ))

        # R13c — alt fragment mein [else] guard hona chahiye (best practice)
        alt_fragments = [
            f for f in fragment_shapes
            if str(f.get("label", "")).strip().lower() == "alt"
        ]
        if alt_fragments:
            all_guard_labels = [
                str(g.get("label", "")).strip().lower()
                for g in guard_shapes
            ]
            has_else = any(
                lbl in ("[else]", "[otherwise]", "[default]")
                for lbl in all_guard_labels
            )
            if not has_else:
                errors.append(ValidationError(
                    error_type  = "ALT_MISSING_ELSE",
                    description = (
                        "ALT fragment mein '[else]' guard nahi hai. "
                        "ALT ke doosre operand mein [else] hona best practice hai."
                    ),
                    suggestion  = (
                        "ALT fragment ke doosre section mein '[else]' guard add karo "
                        "taake default case handle ho."
                    ),
                    severity    = ValidationError.SEVERITY_INFO,
                    element     = "",
                ))

        # ── Score ─────────────────────────────────────────────────────────────
        error_count   = sum(1 for e in errors if e.severity == ValidationError.SEVERITY_ERROR)
        warning_count = sum(1 for e in errors if e.severity == ValidationError.SEVERITY_WARNING)
        score   = max(0, 100 - (error_count * 15) - (warning_count * 5))
        summary = (
            "✅ Sequence Diagram valid hai!"
            if error_count == 0
            else f"❌ {error_count} error(s), {warning_count} warning(s)."
        )
        return self._build_result(errors, score, summary)

    # ═══════════════════════════════════════════════════════════════════════
    # Semantic check (R2)
    # ═══════════════════════════════════════════════════════════════════════

    def _check_interactions_semantic(
        self,
        interactions:     List[Dict],
        diagram_messages: List[str],
    ) -> List[ValidationError]:
        errors: List[ValidationError] = []

        if self._semantic_checker is None:
            try:
                from .semantic_checker import SemanticChecker
                self._semantic_checker = SemanticChecker(threshold=self._threshold)
            except ImportError as e:
                errors.append(ValidationError(
                    error_type  = "SEMANTIC_UNAVAILABLE",
                    description = f"Semantic checking available nahi: {e}",
                    suggestion  = "pip install sentence-transformers chalao.",
                    severity    = ValidationError.SEVERITY_INFO,
                    element     = "",
                ))
                return errors + self._check_interactions_fuzzy(interactions, diagram_messages)

        result = self._semantic_checker.check(
            interactions     = interactions,
            diagram_messages = diagram_messages,
        )

        for sm in result.missing:
            errors.append(ValidationError(
                error_type  = "MISSING_MESSAGE",
                description = (
                    f"Interaction '{sm.interaction_from} → "
                    f"{sm.interaction_message} → {sm.interaction_to}' "
                    f"diagram mein semantically represent nahi hota.\n"
                    f"💡 {sm.reason}"
                ),
                suggestion  = (
                    f"'{sm.interaction_from}' se '{sm.interaction_to}' tak ek arrow "
                    f"draw karo jo '{sm.interaction_message}' ka meaning convey kare."
                ),
                severity    = ValidationError.SEVERITY_WARNING,
                element     = sm.interaction_message,
            ))

        for extra_label in result.extra:
            errors.append(ValidationError(
                error_type  = "EXTRA_MESSAGE",
                description = (
                    f"Arrow '{extra_label}' scenario ke kisi interaction se "
                    f"semantically match nahi karta."
                ),
                suggestion  = (
                    f"Check karo ke '{extra_label}' scenario mein hai ya nahi."
                ),
                severity    = ValidationError.SEVERITY_INFO,
                element     = extra_label,
            ))

        return errors

    def _check_interactions_fuzzy(
        self,
        interactions:     List[Dict],
        diagram_messages: List[str],
    ) -> List[ValidationError]:
        errors = []
        lower  = [m.lower() for m in diagram_messages]
        for i in interactions:
            msg = i.get("message", "")
            frm = i.get("from", "")
            to  = i.get("to",   "")
            if msg and not self._fuzzy_match(msg, lower):
                errors.append(ValidationError(
                    error_type  = "MISSING_MESSAGE",
                    description = f"Interaction '{frm} → {msg} → {to}' diagram mein nahi.",
                    suggestion  = f"'{frm}' se '{to}' ko '{msg}' ke saath arrow draw karo.",
                    severity    = ValidationError.SEVERITY_WARNING,
                    element     = msg,
                ))
        return errors

    # ═══════════════════════════════════════════════════════════════════════
    # Shape category getters  (use DiagramParser keywords)
    # ═══════════════════════════════════════════════════════════════════════

    def _get_lifelines(self, shapes: List[Dict]) -> List[Dict]:
        from .diagram_parser import DiagramParser as DP
        return [s for s in shapes if DP.is_lifeline(s)]

    def _get_actors(self, shapes: List[Dict]) -> List[Dict]:
        """Actor shapes alag se read karo — Flutter ToolType.actor ke liye."""
        from .diagram_parser import DiagramParser as DP
        return [s for s in shapes if DP.is_actor(s)]

    def _get_messages(self, shapes: List[Dict]) -> List[Dict]:
        from .diagram_parser import DiagramParser as DP
        return [s for s in shapes if DP.is_message(s)]

    def _get_activations(self, shapes: List[Dict]) -> List[Dict]:
        from .diagram_parser import DiagramParser as DP
        return [s for s in shapes if DP.is_activation(s)]

    def _get_deletions(self, shapes: List[Dict]) -> List[Dict]:
        from .diagram_parser import DiagramParser as DP
        return [s for s in shapes if DP.is_deletion(s)]

    def _get_fragments(self, shapes: List[Dict]) -> List[Dict]:
        """Alt/opt/loop/par combined fragment boxes."""
        from .diagram_parser import DiagramParser as DP
        return [s for s in shapes if DP.is_fragment(s)]

    def _get_guards(self, shapes: List[Dict]) -> List[Dict]:
        """[condition] guard operand markers inside fragments."""
        from .diagram_parser import DiagramParser as DP
        return [s for s in shapes if DP.is_guard(s)]

    # ═══════════════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _valid_labels(shapes: List[Dict]) -> List[str]:
        return [s["label"] for s in shapes if s.get("label")]

    @staticmethod
    def _filter_participants(raw: List[str]) -> List[str]:
        return [
            t.strip() for t in raw
            if t.strip() and t.strip().lower() not in _NON_PARTICIPANT_TOKENS
        ]

