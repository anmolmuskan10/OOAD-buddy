#!/usr/bin/env bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm --direct-url https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0-py3-none-any.whl
