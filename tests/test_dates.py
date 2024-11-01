import pytest
from redactor import redact_entities_spacy, redact_entities_regex

def test_redact_dates_spacy():
    text = "Her birthday is on March 10, 1985."
    targets = ["dates"]
    stats = {"dates": 0}
    redacted_spans = redact_entities_spacy(text, targets, stats)
    assert stats["dates"] == 1
    assert all(text[start:end] == "March 10, 1985" for start, end in redacted_spans)

def test_redact_dates_regex():
    text = "The meeting is scheduled for 14-08-2020 and 08/14/2020."
    targets = ["dates"]
    stats = {"dates": 0}
    redacted_spans = redact_entities_regex(text, targets, stats)
    assert stats["dates"] == 2
    assert all(text[start:end] in ["14-08-2020", "08/14/2020"] for start, end in redacted_spans)
