import pytest
from redactor import redact_entities_spacy, redact_entities_hf, redact_entities_regex

def test_redact_person_names_spacy():
    text = "Alice Johnson went to the market."
    targets = ["names"]
    stats = {"names": 0}
    redacted_spans = redact_entities_spacy(text, targets, stats)
    assert stats["names"] == 1
    assert all(text[start:end] == "Alice Johnson" for start, end in redacted_spans)

def test_redact_person_names_hf():
    text = "Contact Sarah Connor via email."
    targets = ["names"]
    stats = {"names": 0}
    redacted_spans = redact_entities_hf(text, targets, stats)
    assert stats["names"] == 1
    assert all(text[start:end] == "Sarah Connor" for start, end in redacted_spans)

def test_redact_person_names_regex():
    text = "Meet John Doe and Jane Smith."
    targets = ["names"]
    stats = {"names": 0}
    redacted_spans = redact_entities_regex(text, targets, stats)
    assert stats["names"] == 2
