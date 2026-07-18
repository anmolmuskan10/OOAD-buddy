"""
OOAD Diagram Validation Engine - Flask Backend
Hybrid Validation: Gemini AI (primary) + Rule-Based (fallback)
All 3 diagram types: class, usecase, sequence

FIX: Image upload se diagram_type auto-detect via Gemini Vision.
     Agar diagram_type missing/unknown ho toh image analyse karke
     automatically pata lagta hai ke class/usecase/sequence hai.

UPDATE: gpt-4o -> gpt-4o-mini (rate limit fix), timeout 120s, retry logic added
"""

import os
import re 
import json
import base64
import logging
import time
import urllib.request
import urllib.error
from flask import Flask, request, jsonify
from flask_cors import CORS

from nlp_extractor import NLPExtractor
from validators.class_validator import ClassDiagramValidator
from validators.usecase_validator import UseCaseValidator
from validators.sequence_validator import SequenceDiagramValidator
from validators.openai_validator import validate_with_openai, validate_with_openai_image

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

app       = Flask(__name__)
CORS(app)
extractor = NLPExtractor()

# OpenAI API config
_OPENAI_API_BASE = "https://api.openai.com/v1"
_VISION_MODELS   = ["gpt-4o-mini"]   # FIX: gpt-4o-mini use karo (sasta + zyada rate limit)
_TIMEOUT         = 120               # FIX: 60 se badha ke 120 kiya (worker crash band)
_RETRY_WAIT      = 65                # Rate limit pe wait seconds


def _get_api_key():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    return key if key else None


# ─────────────────────────────────────────────────────────────────────────────
#  AUTO-DETECT diagram type from image using OpenAI Vision
# ─────────────────────────────────────────────────────────────────────────────

def _detect_diagram_type_from_image(image_b64: str, mime_type: str = "image/png") -> str:
    """
    OpenAI Vision se image analyse karke diagram type detect karo.
    Returns: 'class' | 'usecase' | 'sequence'
    Default: 'class' (agar detect na ho sake)
    """
    api_key = _get_api_key()
    if not api_key:
        _log.warning("OPENAI_API_KEY missing - cannot auto-detect diagram type")
        return "class"

    prompt = """Look at this UML diagram image carefully.

Determine which ONE of these three diagram types it is:
1. CLASS diagram     - has rectangles with class names, attributes, methods; arrows for inheritance/association
2. USE CASE diagram  - has stick figures (actors), ovals/ellipses (use cases), system boundary rectangle
3. SEQUENCE diagram  - has vertical dashed lines (lifelines), horizontal arrows between them (messages)

Reply with ONLY one word - exactly one of: class, usecase, sequence
Do not explain. Just the single word."""

    payload = json.dumps({
        "model": "gpt-4o-mini",   # FIX: gpt-4o-mini use karo
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
            ],
        }],
        "temperature": 0.0,
        "max_tokens": 10,
    }).encode("utf-8")

    for model in _VISION_MODELS:
        url = f"{_OPENAI_API_BASE}/chat/completions"
        req = urllib.request.Request(
            url, data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        # FIX: Retry logic - rate limit pe dobara try karo
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                text = body["choices"][0]["message"]["content"].strip().lower()
                text = re.sub(r"[^a-z]", "", text.split()[0] if text.split() else "")
                if text in ("class", "usecase", "sequence"):
                    _log.info("Auto-detected diagram type: '%s' (model: %s)", text, model)
                    return text
                if "class" in text:   return "class"
                if "use" in text:     return "usecase"
                if "seq" in text:     return "sequence"
                break  # response mila, loop band karo
            except urllib.error.HTTPError as e:
                if e.code == 429:  # Rate limit error
                    if attempt < 2:
                        _log.warning("Rate limit hit (attempt %d/3) - waiting %ds...", attempt + 1, _RETRY_WAIT)
                        time.sleep(_RETRY_WAIT)
                    else:
                        _log.error("Rate limit - 3 attempts failed for model %s", model)
                else:
                    _log.warning("Vision model %s HTTP error %d: %s", model, e.code, e)
                    break
            except Exception as e:
                _log.warning("Vision model %s failed: %s", model, e)
                break

    _log.warning("Could not auto-detect diagram type - defaulting to 'class'")
    return "class"


# ─────────────────────────────────────────────────────────────────────────────
#  Normalize diagram_type string
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
#  MERGE FIX (2026-07): error types the OpenAI prompt explicitly tells the
#  model to SKIP because "the rule system handles it" — but app.py never
#  actually ran the rule system when OpenAI succeeded, so these were being
#  silently dropped 100% of the time. This map lists, per diagram type,
#  which error_types must ALWAYS be sourced from the rule-based validator,
#  even when OpenAI is the primary engine.
# ─────────────────────────────────────────────────────────────────────────────
_RULE_ONLY_ERROR_TYPES = {
    "class":    {"WRONG_MULTIPLICITY", "MISSING_MULTIPLICITY", "INVALID_MULTIPLICITY",
                 "MISSING_ASSOCIATION_NAME"},
    "usecase":  {
        "DISCONNECTED_ACTOR", "ISOLATED_USE_CASE",
        "DUPLICATE_ACTOR", "DUPLICATE_USE_CASE",
        "UNLABELLED_ACTOR", "UNLABELLED_USE_CASE",
        "USE_CASE_TOO_VAGUE",
    },
    "sequence": set(),
}


def _merge_rule_only_checks(final_result: dict, rule_validator, extracted: dict,
                             shapes: list, dtype: str) -> dict:
    """
    Run the rule-based validator alongside an OpenAI result and merge in
    ONLY the error types that OpenAI was told to skip. Requires shape/geometry
    data to be meaningful (image-only requests have no shapes, so this is a
    no-op there — those checks simply can't run without geometry).
    """
    wanted = _RULE_ONLY_ERROR_TYPES.get(dtype, set())
    if not wanted or not shapes:
        return final_result

    try:
        rule_result = rule_validator.validate(extracted, shapes)
    except Exception as e:
        _log.warning("Rule-only merge: rule_validator failed: %s", e)
        return final_result

    already_present = {
        (str(i.get("error_type", "")), str(i.get("element", "")))
        for bucket in ("errors", "warnings", "info")
        for i in final_result.get(bucket, [])
    }
    # Multiplicity errors: also track by normalized (from,to) pair, since the
    # LLM's `element` string and the rule engine's `element` string ("A → B")
    # aren't formatted the same way and would otherwise both get kept.
    _mult_types = {"MISSING_MULTIPLICITY", "WRONG_MULTIPLICITY", "INVALID_MULTIPLICITY"}
    already_mult_pairs = set()
    for bucket in ("errors", "warnings", "info"):
        for i in final_result.get(bucket, []):
            if i.get("error_type") in _mult_types:
                fix = i.get("auto_fix") or {}
                frm = str(fix.get("from_element", "")).strip().lower()
                to  = str(fix.get("to_element", "")).strip().lower()
                if frm and to:
                    already_mult_pairs.add(frozenset({frm, to}))

    added_any = False
    for bucket in ("errors", "warnings", "info"):
        for item in rule_result.get(bucket, []):
            et = item.get("error_type")
            if et not in wanted:
                continue
            key = (str(et), str(item.get("element", "")))
            if key in already_present:
                continue
            if et in _mult_types:
                elem = str(item.get("element", ""))
                parts = [p.strip().lower() for p in elem.split("→")]
                if len(parts) == 2 and frozenset(parts) in already_mult_pairs:
                    continue
            item.setdefault("source", "rule")
            final_result.setdefault(bucket, []).append(item)
            already_present.add(key)
            added_any = True

    if added_any:
        final_result["is_valid"] = len(final_result.get("errors", [])) == 0
        final_result["total_issues"] = (
            len(final_result.get("errors", [])) +
            len(final_result.get("warnings", [])) +
            len(final_result.get("info", []))
        )
        final_result["fixable_count"] = sum(
            1 for bucket in ("errors", "warnings", "info")
            for i in final_result.get(bucket, [])
            if i.get("auto_fix", {}).get("fixable")
        )

    return final_result


def _normalize_dtype(raw: str) -> str:
    """'UseCase', 'use_case', 'CLASS' etc. -> 'class'/'usecase'/'sequence' or ''"""
    s = raw.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    if s in ("class", "classdiagram"):          return "class"
    if s in ("usecase", "usecasediagram", "uc"): return "usecase"
    if s in ("sequence", "sequencediagram", "seq"): return "sequence"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "OOAD Hybrid Validation Engine",
        "status":  "running",
        "mode":    "OpenAI GPT-4o-mini (primary) + Rule-Based (fallback)",
        "features": {
            "image_auto_detect": "Upload image -> OpenAI Vision auto-detects diagram type",
            "diagram_types":     ["class", "usecase", "sequence"],
        },
        "endpoints": {"/health": "health check", "/validate": "validate diagram", "/extract": "NLP only"}
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status":         "ok",
        "message":        "Hybrid Validation Engine is running",
        "openai_enabled": bool(_get_api_key()),
    })


@app.route('/validate', methods=['POST'])
def validate():
    """
    Format A - JSON body:
        { "scenario": "...", "diagram_type": "class|usecase|sequence", "shapes": [...] }

    Format B - multipart form + image file:
        scenario      = text field
        diagram_type  = optional (auto-detected from image if missing)
        image         = image file (PNG/JPG)
        shapes        = optional JSON string
    """
    image_b64  = None
    mime_type  = "image/png"
    shapes     = []
    scenario   = ""
    dtype_raw  = ""

    if request.content_type and "multipart" in request.content_type:
        # Format B: form data + image
        scenario   = (request.form.get("scenario", "") or "").strip()
        dtype_raw  = (request.form.get("diagram_type", "") or "").strip()
        shapes_str = request.form.get("shapes", "")
        if shapes_str:
            try:    shapes = json.loads(shapes_str)
            except: shapes = []

        img_file = request.files.get("image")
        if img_file:
            image_b64 = base64.b64encode(img_file.read()).decode("utf-8")
            mime_type = img_file.mimetype or "image/png"
    else:
        # Format A: JSON body
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "No JSON body received"}), 400
        scenario  = (data.get("scenario", "") or "").strip()
        dtype_raw = (data.get("diagram_type", "") or "").strip()
        shapes    = data.get("shapes", [])
        # Also support base64 image inside JSON
        if data.get("image"):
            image_b64 = data["image"]
            mime_type = data.get("mime_type", "image/png")

    if not scenario:
        return jsonify({"error": "scenario field is required"}), 400

    # ── Determine diagram type ─────────────────────────────────────────────
    dtype = _normalize_dtype(dtype_raw)

    if not dtype:
        if image_b64:
            _log.info("diagram_type missing - auto-detecting from image...")
            dtype = _detect_diagram_type_from_image(image_b64, mime_type)
        else:
            return jsonify({
                "error": (
                    f"Unknown or missing diagram_type: '{dtype_raw}'. "
                    "Use 'class', 'usecase', or 'sequence'. "
                    "Or upload an image for auto-detection."
                )
            }), 400

    # Select rule-based validator
    if dtype == "usecase":
        rule_validator = UseCaseValidator()
    elif dtype == "sequence":
        rule_validator = SequenceDiagramValidator()
    else:
        dtype = "class"
        rule_validator = ClassDiagramValidator()

    # ── NLP extraction ─────────────────────────────────────────────────────
    extracted = extractor.extract(scenario)

    # ── OpenAI AI first (PRIMARY) ──────────────────────────────────────────
    # Jab image available ho → OpenAI Vision use karo (image directly analyze)
    # Jab sirf shapes hon → text-based OpenAI use karo
    if image_b64:
        _log.info("Image available — using OpenAI Vision for '%s' diagram", dtype)
        gemini_result = validate_with_openai_image(
            scenario=scenario,
            image_b64=image_b64,
            mime_type=mime_type,
            diagram_type=dtype,
        )
    else:
        gemini_result = validate_with_openai(scenario, shapes, diagram_type=dtype)

    if gemini_result:
        _log.info("OpenAI validation used for '%s' diagram", dtype)
        gemini_result["validation_mode"] = "openai"
        final_result = gemini_result
        # FIX: OpenAI's prompt explicitly skips certain error types (e.g.
        # WRONG_MULTIPLICITY, DISCONNECTED_ACTOR, ISOLATED_USE_CASE...)
        # assuming the rule-based validator reports them instead. Since the
        # rule-based validator was never actually invoked in this branch,
        # those checks were silently missing. Merge them back in here.
        final_result = _merge_rule_only_checks(
            final_result, rule_validator, extracted, shapes, dtype
        )
    else:
        # Fallback to rule-based
        _log.warning("OpenAI unavailable - rule-based fallback for '%s'", dtype)
        rule_result = rule_validator.validate(extracted, shapes)
        rule_result["validation_mode"] = "rule-based (OpenAI unavailable)"
        # Rule-based has no auto_fix — set fixable_count to 0
        rule_result.setdefault("fixable_count", 0)
        final_result = rule_result

    return jsonify({
        "diagram_type":        dtype,
        "auto_detected":       bool(image_b64 and not _normalize_dtype(dtype_raw)),
        "extracted_elements":  extracted,
        "validation_result":   final_result,
    })


@app.route('/extract', methods=['POST'])
def extract_only():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body received"}), 400
    scenario = (data.get("scenario", "") or "").strip()
    if not scenario:
        return jsonify({"error": "scenario required"}), 400
    return jsonify(extractor.extract(scenario))


if __name__ == '__main__':
    key = _get_api_key()
    if not key:
        _log.warning("OPENAI_API_KEY not set - rule-based only. Image auto-detect DISABLED.")
    else:
        _log.info("OpenAI API key found - AI validation + image auto-detect ENABLED")
    app.run(debug=True, host='0.0.0.0', port=5000)
