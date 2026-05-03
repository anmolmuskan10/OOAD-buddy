#!/usr/bin/env bash
set -e
pip install --no-cache-dir -r requirements.txt
python -m spacy download en_core_web_sm
