"""
validators/semantic_checker.py
────────────────────────────────
Offline semantic matching using sentence-transformers.

Koi API nahi, koi cost nahi, koi internet nahi — sab local.

Model:  all-MiniLM-L6-v2
  - Fast (6x faster than large models)
  - Accurate enough for short UML labels
  - Size: ~80MB (ek baar download, phir cache mein)
  - Free, open-source (Apache 2.0)

Installation (ek baar):
    pip install sentence-transformers

How it works:
    "Find user"          → [0.82, 0.31, -0.14, ...]  (384-dim vector)
    "query user record"  → [0.79, 0.28, -0.11, ...]
    cosine_similarity    → 0.89  ✅  (threshold: 0.45)

    "Find user"          → [0.82, 0.31, ...]
    "send email"         → [0.11, -0.42, ...]
    cosine_similarity    → 0.12  ❌  (below threshold)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# ── Lazy import guard ─────────────────────────────────────────────────────────
# sentence-transformers heavy import — only load when actually needed
_model_cache = None

def _get_model():
    """Load model once, cache it for all subsequent calls."""
    global _model_cache
    if _model_cache is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            _model_cache = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            raise ImportError(
                "sentence-transformers install nahi hai.\n"
                "Chalao:  pip install sentence-transformers"
            )
    return _model_cache


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SemanticMatch:
    """Ek interaction ka result."""
    interaction_from:    str
    interaction_message: str
    interaction_to:      str
    diagram_label:       str    # best matching label, "" if none found
    matched:             bool
    similarity:          float  # 0.0 – 1.0
    reason:              str    # human-readable explanation


@dataclass
class SemanticResult:
    """Poori validation ka result."""
    matched:  List[SemanticMatch] = field(default_factory=list)   # ✅ found in diagram
    missing:  List[SemanticMatch] = field(default_factory=list)   # ❌ not in diagram
    extra:    List[str]           = field(default_factory=list)   # diagram mein hai, scenario mein nahi


# ── Main checker ──────────────────────────────────────────────────────────────

class SemanticChecker:
    """
    Sentence-transformer based offline semantic checker.

    Parameters
    ----------
    threshold : float
        Cosine similarity cutoff.  Default 0.45 works well for short UML labels.
        Lower  → more lenient  (fewer false errors, might miss real mistakes)
        Higher → more strict   (more errors, might flag valid diagrams)
    """

    def __init__(self, threshold: float = 0.45) -> None:
        self.threshold = threshold

    # ── Public method ─────────────────────────────────────────────────────────

    def check(
        self,
        interactions:     List[dict],   # [{"from":…, "message":…, "to":…}]
        diagram_messages: List[str],    # diagram ke arrow labels
    ) -> SemanticResult:
        """
        Har scenario interaction ke liye best matching diagram label dhundho.

        Returns SemanticResult with matched / missing / extra.
        """
        result = SemanticResult()

        if not interactions:
            # Koi interaction nahi → sab diagram labels extra hain
            result.extra = list(diagram_messages)
            return result

        if not diagram_messages:
            # Diagram mein koi label nahi → sab interactions missing hain
            for i in interactions:
                result.missing.append(SemanticMatch(
                    interaction_from    = i.get("from", ""),
                    interaction_message = i.get("message", ""),
                    interaction_to      = i.get("to", ""),
                    diagram_label       = "",
                    matched             = False,
                    similarity          = 0.0,
                    reason              = "Diagram mein koi message arrow nahi hai.",
                ))
            return result

        model = _get_model()

        # ── Build query strings from interactions ─────────────────────────────
        # Full context dene se similarity better hoti hai
        # e.g.  "User sends login to App"  vs just  "login"
        interaction_queries = [
            self._interaction_to_text(i) for i in interactions
        ]

        # ── Encode all at once (batch = fast) ────────────────────────────────
        import numpy as np

        query_embeddings   = model.encode(interaction_queries,   convert_to_numpy=True)
        diagram_embeddings = model.encode(diagram_messages,      convert_to_numpy=True)

        # ── Cosine similarity matrix: shape (n_interactions, n_diagram_labels) ─
        # cos_sim[i][j] = similarity between interaction i and diagram label j
        cos_sim = self._cosine_similarity_matrix(query_embeddings, diagram_embeddings)

        # ── Match each interaction to best diagram label ───────────────────────
        matched_diagram_indices = set()

        for i_idx, interaction in enumerate(interactions):
            similarities   = cos_sim[i_idx]           # scores for all diagram labels
            best_j         = int(np.argmax(similarities))
            best_score     = float(similarities[best_j])
            best_label     = diagram_messages[best_j]

            frm = interaction.get("from",    "")
            msg = interaction.get("message", "")
            to  = interaction.get("to",      "")

            if best_score >= self.threshold:
                matched_diagram_indices.add(best_j)
                sm = SemanticMatch(
                    interaction_from    = frm,
                    interaction_message = msg,
                    interaction_to      = to,
                    diagram_label       = best_label,
                    matched             = True,
                    similarity          = round(best_score, 3),
                    reason              = (
                        f"'{best_label}' semantically '{msg}' se match karta hai "
                        f"(similarity: {best_score:.0%})."
                    ),
                )
                result.matched.append(sm)
            else:
                sm = SemanticMatch(
                    interaction_from    = frm,
                    interaction_message = msg,
                    interaction_to      = to,
                    diagram_label       = best_label,   # closest we found, still below threshold
                    matched             = False,
                    similarity          = round(best_score, 3),
                    reason              = (
                        f"Koi diagram label '{msg}' ka meaning convey nahi karta. "
                        f"Closest tha '{best_label}' (similarity: {best_score:.0%}) "
                        f"— threshold {self.threshold:.0%} se kam hai."
                    ),
                )
                result.missing.append(sm)

        # ── Find extra diagram labels (not matched to any interaction) ─────────
        for j_idx, label in enumerate(diagram_messages):
            if j_idx not in matched_diagram_indices:
                result.extra.append(label)

        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _interaction_to_text(interaction: dict) -> str:
        """
        Interaction dict ko ek natural sentence mein convert karo.
        This gives the model more context than just the message verb.

        e.g. {"from":"App", "message":"query user", "to":"Database"}
             → "App query user Database"
        """
        frm = interaction.get("from",    "")
        msg = interaction.get("message", "")
        to  = interaction.get("to",      "")
        parts = [p for p in [frm, msg, to] if p]
        return " ".join(parts)

    @staticmethod
    def _cosine_similarity_matrix(a, b):
        """
        Compute cosine similarity between every row in `a` and every row in `b`.
        Returns matrix of shape (len(a), len(b)).
        Pure numpy — no extra dependencies.
        """
        import numpy as np
        # Normalize rows to unit vectors
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        return np.dot(a_norm, b_norm.T)
