import unittest
from redactor import identify_concept_sentences

class TestIdentifyConceptSentences(unittest.TestCase):
    def test_no_concepts(self):
        text = "This is a test. No sensitive information here."
        concepts = ["confidential", "proprietary"]
        expected = []
        result = identify_concept_sentences(text, concepts)
        self.assertEqual(result, expected)
    
    def test_single_concept(self):
        text = "This is a confidential document.\nAnother sentence."
        concepts = ["confidential"]
        expected = [(0, 32)]
        result = identify_concept_sentences(text, concepts)
        self.assertEqual(result, expected)
    
    def test_multiple_concepts(self):
        text = (
            "This document is confidential.\n"
            "It contains proprietary information.\n"
            "Public information is also included."
        )
        concepts = ["confidential", "proprietary"]
        expected = [(0, 30), (30, 67)]
        result = identify_concept_sentences(text, concepts)
        self.assertEqual(result, expected)
    
    def test_concepts_case_insensitive(self):
        text = "This is a Confidential document.\nAnother sentence with Proprietary data."
        concepts = ["confidential", "proprietary"]
        expected = [(0, 32), (32, 72)]
        result = identify_concept_sentences(text, concepts)
        self.assertEqual(result, expected)
    
    def test_concepts_partial_match(self):
        text = "This is a confidant document.\nProprietarily speaking."
        concepts = ["confidential", "proprietary"]
        expected = []
        result = identify_concept_sentences(text, concepts)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
