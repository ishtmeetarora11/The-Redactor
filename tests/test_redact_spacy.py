import unittest
from unittest.mock import patch, Mock
from redactor import redact_entities_spacy

class TestRedactEntitiesSpacy(unittest.TestCase):
    def setUp(self):
        self.text = "John Doe was born on January 1, 1990. He lives in New York."
        self.targets = ['names', 'dates', 'addresses']
        self.stats = {'names': 0, 'dates': 0, 'phones': 0, 'addresses': 0, 'concepts': 0}
    
    @patch('redactor.nlp')
    def test_redact_names_dates_addresses(self, mock_nlp):
        # Mock SpaCy's doc with entities
        mock_doc = Mock()
        mock_doc.ents = [
            Mock(label_='PERSON', start_char=0, end_char=8),
            Mock(label_='DATE', start_char=18, end_char=32),
            Mock(label_='GPE', start_char=42, end_char=50)
        ]
        mock_nlp.return_value = mock_doc
        
        expected_spans = [(0, 8), (18, 32), (42, 50)]
        result = redact_entities_spacy(self.text, self.targets, self.stats)
        self.assertEqual(result, expected_spans)
        self.assertEqual(self.stats, {'names': 1, 'dates': 1, 'phones': 0, 'addresses': 1, 'concepts': 0})
    
    @patch('redactor.nlp')
    def test_non_target_entities(self, mock_nlp):
        mock_doc = Mock()
        mock_doc.ents = [
            Mock(label_='ORG', start_char=0, end_char=8),
            Mock(label_='EVENT', start_char=18, end_char=32)
        ]
        mock_nlp.return_value = mock_doc
        
        expected_spans = []
        result = redact_entities_spacy(self.text, self.targets, self.stats)
        self.assertEqual(result, expected_spans)
        self.assertEqual(self.stats, {'names': 0, 'dates': 0, 'phones': 0, 'addresses': 0, 'concepts': 0})
    
    @patch('redactor.nlp')
    def test_partial_targets(self, mock_nlp):
        # Only redact 'names' and 'addresses'
        mock_doc = Mock()
        mock_doc.ents = [
            Mock(label_='PERSON', start_char=0, end_char=8),
            Mock(label_='DATE', start_char=18, end_char=32),
            Mock(label_='GPE', start_char=42, end_char=50)
        ]
        mock_nlp.return_value = mock_doc
        targets = ['names', 'addresses']
        expected_spans = [(0, 8), (42, 50)]
        result = redact_entities_spacy(self.text, targets, self.stats)
        self.assertEqual(result, expected_spans)
        self.assertEqual(self.stats, {'names': 1, 'dates': 0, 'phones': 0, 'addresses': 1, 'concepts': 0})

if __name__ == '__main__':
    unittest.main()
