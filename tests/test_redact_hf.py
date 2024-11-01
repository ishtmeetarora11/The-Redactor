import unittest
from unittest.mock import patch
from redactor import redact_entities_hf

class TestRedactEntitiesHF(unittest.TestCase):
    def setUp(self):
        self.text = "Jane Smith visited Berlin on 05/20/2021."
        self.targets = ['names', 'addresses']
        self.stats = {'names': 0, 'dates': 0, 'phones': 0, 'addresses': 0, 'concepts': 0}
    
    @patch('redactor.hf_ner_pipeline')
    def test_redact_names_addresses(self, mock_pipeline):
        # Mock Hugging Face NER results
        mock_pipeline.return_value = [
            {'entity_group': 'PER', 'score': 0.99, 'word': 'Jane Smith', 'start': 0, 'end': 10},
            {'entity_group': 'LOC', 'score': 0.98, 'word': 'Berlin', 'start': 19, 'end': 25}
        ]
        
        expected_spans = [(0, 10), (19, 25)]
        result = redact_entities_hf(self.text, self.targets, self.stats)
        self.assertEqual(result, expected_spans)
        self.assertEqual(self.stats, {'names': 1, 'dates': 0, 'phones': 0, 'addresses': 1, 'concepts': 0})
    
    @patch('redactor.hf_ner_pipeline')
    def test_non_target_entities(self, mock_pipeline):
        mock_pipeline.return_value = [
            {'entity_group': 'ORG', 'score': 0.95, 'word': 'OpenAI', 'start': 0, 'end': 6},
            {'entity_group': 'MISC', 'score': 0.90, 'word': 'GPT-4', 'start': 15, 'end': 20}
        ]
        
        expected_spans = []
        result = redact_entities_hf(self.text, self.targets, self.stats)
        self.assertEqual(result, expected_spans)
        self.assertEqual(self.stats, {'names': 0, 'dates': 0, 'phones': 0, 'addresses': 0, 'concepts': 0})
    
    @patch('redactor.hf_ner_pipeline')
    def test_partial_targets(self, mock_pipeline):
        # Only redact 'addresses'
        mock_pipeline.return_value = [
            {'entity_group': 'PER', 'score': 0.99, 'word': 'Jane Smith', 'start': 0, 'end': 10},
            {'entity_group': 'LOC', 'score': 0.98, 'word': 'Berlin', 'start': 19, 'end': 25}
        ]
        targets = ['addresses']
        expected_spans = [(19, 25)]
        result = redact_entities_hf(self.text, targets, self.stats)
        self.assertEqual(result, expected_spans)
        self.assertEqual(self.stats, {'names': 0, 'dates': 0, 'phones': 0, 'addresses': 1, 'concepts': 0})

if __name__ == '__main__':
    unittest.main()
