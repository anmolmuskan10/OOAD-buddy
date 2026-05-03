"""
OOAD Diagram Validation Engine - Flask Backend
Hybrid Validation: Gemini AI (primary) + Rule-Based (fallback)
All 3 diagram types: class, usecase, sequence

FIX: Image upload se diagram_type auto-detect via Gemini Vision.
     Agar diagram_type missing/unknown ho toh image analyse karke
     automatically pata lagta hai ke class/usecase/sequence hai.
"""

import os
import re
import json
import base64
import logging
import urllib.request
import urllib.error
from flask import Flask, request, jsonify
from flask_cors import CORS

from nlp_extractor import NLPExtractor
from validators.class_validator import ClassDiagramValidator
from validators.usecase_validator import UseCaseValidator
from validators.sequence_validator import SequenceDiagramValidator
from validators.gemini_validator import validate_with_gemini, validate_with_gemini_image

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

app       = Flask(__name__)
CORS(app)
extractor = NLPExtractor()

# Gemini API config
_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
_VISION_MODELS   = ["gemini-2.5-flash-lite", "gemini-2.0-flash-lite", "gemini-2.0-flash"]
_TIMEOUT         = 60


def _get_api_key():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    return key if key else None


# ─────────────────────────────────────────────────────────────────────────────
#  AUTO-DETECT diagram type from image using Gemini Vision
# ─────────────────────────────────────────────────────────────────────────────

def _detect_diagram_type_from_image(image_b64: str, mime_type: str = "image/png") -> str:
    """
    Gemini Vision se image analyse karke diagram type detect karo.
    Returns: 'class' | 'usecase' | 'sequence'
    Default: 'class' (agar detect na ho sake)
    """
    api_key = _get_api_key()
    if not api_key:
        _log.warning("GEMINI_API_KEY missing - cannot auto-detect diagram type")
        return "class"

    prompt = """Look at this UML diagram image carefully.

Determine which ONE of these three diagram types it is:
1. CLASS diagram     - has rectangles with class names, attributes, methods; arrows for inheritance/association
2. USE CASE diagram  - has stick figures (actors), ovals/ellipses (use cases), system boundary rectangle
3. SEQUENCE diagram  - has vertical dashed lines (lifelines), horizontal arrows between them (messages)

Reply with ONLY one word - exactly one of: class, usecase, sequence
Do not explain. Just the single word."""

    payload = json.dumps({
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime_type, "data": image_b64}}
            ]
        }],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 10}
    }).encode("utf-8")

    for model in _VISION_MODELS:
        url = f"{_GEMINI_API_BASE}/{model}:generateContent?key={api_key}"
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            text = body["candidates"][0]["content"]["parts"][0]["text"].strip().lower()
            text = re.sub(r"[^a-z]", "", text.split()[0] if text.split() else "")
            if text in ("class", "usecase", "sequence"):
                _log.info("Auto-detected diagram type: '%s' (model: %s)", text, model)
                return text
            if "class" in text:   return "class"
            if "use" in text:     return "usecase"
            if "seq" in text:     return "sequence"
        except Exception as e:
            _log.warning("Vision model %s failed: %s", model, e)

    _log.warning("Could not auto-detect diagram type - defaulting to 'class'")
    return "class"


# ─────────────────────────────────────────────────────────────────────────────
#  Normalize diagram_type string
# ─────────────────────────────────────────────────────────────────────────────

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
        "mode":    "Gemini AI (primary) + Rule-Based (fallback)",
        "features": {
            "image_auto_detect": "Upload image -> Gemini Vision auto-detects diagram type",
            "diagram_types":     ["class", "usecase", "sequence"],
        },
        "endpoints": {"/health": "health check", "/validate": "validate diagram", "/extract": "NLP only"}
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status":         "ok",
        "message":        "Hybrid Validation Engine is running",
        "gemini_enabled": bool(_get_api_key()),
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

    # ── Gemini AI first (PRIMARY) ──────────────────────────────────────────
    # Jab image available ho → Gemini Vision use karo (image directly analyze)
    # Jab sirf shapes hon → text-based Gemini use karo
    if image_b64:
        _log.info("Image available — using Gemini Vision for '%s' diagram", dtype)
        gemini_result = validate_with_gemini_image(
            scenario=scenario,
            image_b64=image_b64,
            mime_type=mime_type,
            diagram_type=dtype,
        )
    else:
        gemini_result = validate_with_gemini(scenario, shapes, diagram_type=dtype)

    if gemini_result:
        _log.info("Gemini validation used for '%s' diagram", dtype)
        gemini_result["validation_mode"] = "gemini"
        final_result = gemini_result
    else:
        # Fallback to rule-based
        _log.warning("Gemini unavailable - rule-based fallback for '%s'", dtype)
        rule_result = rule_validator.validate(extracted, shapes)
        rule_result["validation_mode"] = "rule-based (Gemini unavailable)"
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
        _log.warning("GEMINI_API_KEY not set - rule-based only. Image auto-detect DISABLED.")
    else:
        _log.info("Gemini API key found - AI validation + image auto-detect ENABLED")
    app.run(debug=True, host='0.0.0.0', port=5000)
