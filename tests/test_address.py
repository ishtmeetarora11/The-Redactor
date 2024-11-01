import pytest
from redactor import redact_entities_spacy, redact_entities_regex

def test_redact_address_spacy():
    text = "I live at 456 Maple Avenue."
    targets = ["addresses"]
    stats = {"addresses": 0}
    redacted_spans = redact_entities_spacy(text, targets, stats)
    assert stats["addresses"] == 0

def test_redact_address_regex():
    text = "Visit us at 789 Elm St., Springfield, IL 62704."
    targets = ["addresses"]
    stats = {"addresses": 0}
    redacted_spans = redact_entities_regex(text, targets, stats)
    assert stats["addresses"] == 1
