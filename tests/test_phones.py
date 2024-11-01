import pytest
from redactor import redact_entities_spacy, redact_entities_regex

def test_redact_phone_numbers_spacy():
    text = "Call me at 123-456-7890."
    targets = ["phones"]
    stats = {"phones": 0}
    redacted_spans = redact_entities_spacy(text, targets, stats)
    assert stats["phones"] == 1
    assert all(text[start:end] == "123-456-7890" for start, end in redacted_spans)

def test_redact_phone_numbers_regex():
    text = "Contact me at (123) 456-7890 or 987.654.3210."
    targets = ["phones"]
    stats = {"phones": 0}
    redacted_spans = redact_entities_regex(text, targets, stats)
    assert stats["phones"] == 2
