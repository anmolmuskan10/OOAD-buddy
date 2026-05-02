"""
OOAD Diagram Validation Engine - Flask Backend
Hybrid Validation: Gemini AI (primary) + Rule-Based (fallback)
All 3 diagram types: class, usecase, sequence
"""

import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

from nlp_extractor import NLPExtractor
from validators.class_validator import ClassDiagramValidator
from validators.usecase_validator import UseCaseValidator
from validators.sequence_validator import SequenceDiagramValidator
from validators.gemini_validator import validate_with_gemini

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

app       = Flask(__name__)
CORS(app)
extractor = NLPExtractor()


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "OOAD Hybrid Validation Engine",
        "status":  "running",
        "mode":    "Gemini AI + Rule-Based fallback",
        "endpoints": {"/health": "health check", "/validate": "validate diagram", "/extract": "NLP extract only"}
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status":         "ok",
        "message":        "Hybrid Validation Engine is running",
        "gemini_enabled": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
    })


@app.route('/validate', methods=['POST'])
def validate():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body received"}), 400

    scenario     = data.get("scenario",     "").strip()
    diagram_type = data.get("diagram_type", "").strip().lower()
    shapes       = data.get("shapes",       [])

    if not scenario:
        return jsonify({"error": "scenario field is required"}), 400
    if not diagram_type:
        return jsonify({"error": "diagram_type field is required"}), 400

    # ── Step 1: NLP extraction ─────────────────────────────────────────────
    extracted = extractor.extract(scenario)

    # ── Step 2: Determine diagram type ────────────────────────────────────
    if "usecase" in diagram_type or "use_case" in diagram_type:
        dtype       = "usecase"
        rule_validator = UseCaseValidator()
    elif "class" in diagram_type:
        dtype       = "class"
        rule_validator = ClassDiagramValidator()
    elif "sequence" in diagram_type:
        dtype       = "sequence"
        rule_validator = SequenceDiagramValidator()
    else:
        return jsonify({"error": f"Unknown diagram_type: {diagram_type}"}), 400

    # ── Step 3: Try Gemini AI first ────────────────────────────────────────
    gemini_result = validate_with_gemini(scenario, shapes, diagram_type=dtype)

    if gemini_result:
        _log.info("Gemini validation successful for %s diagram", dtype)
        gemini_result["validation_mode"] = "gemini"
        final_result = gemini_result
    else:
        # ── Step 4: Fallback to rule-based ─────────────────────────────────
        _log.warning("Gemini unavailable — using rule-based only for %s", dtype)
        rule_result = rule_validator.validate(extracted, shapes)
        rule_result["validation_mode"] = "rule-based (Gemini unavailable)"
        final_result = rule_result

    return jsonify({
        "diagram_type":       diagram_type,
        "extracted_elements": extracted,
        "validation_result":  final_result,
    })


@app.route('/extract', methods=['POST'])
def extract_only():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body received"}), 400
    scenario = data.get("scenario", "").strip()
    if not scenario:
        return jsonify({"error": "scenario required"}), 400
    return jsonify(extractor.extract(scenario))


if __name__ == '__main__':
    if not os.environ.get("GEMINI_API_KEY"):
        _log.warning("⚠️  GEMINI_API_KEY not set — rule-based fallback only")
    else:
        _log.info("✅ Gemini API key detected — AI validation enabled for all diagram types")
    app.run(debug=True, host='0.0.0.0', port=5000)
