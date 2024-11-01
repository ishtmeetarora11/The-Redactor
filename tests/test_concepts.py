import pytest
from redactor import identify_concept_sentences

def test_identify_concept_sentences():
    text = "Climate change is an important issue. We must address it urgently."
    concepts = ["climate change", "urgent"]
    concept_spans = identify_concept_sentences(text, concepts)
    assert len(concept_spans) == 1
    assert all(text[start:end].strip() in ["Climate change is an important issue.", "We must address it urgently."] for start, end in concept_spans)
