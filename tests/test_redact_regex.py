import unittest
from redactor import redact_entities_regex

class TestRedactEntitiesRegex(unittest.TestCase):
    def setUp(self):
        self.text = (
            "Contact John Doe at john.doe@example.com or (123) 456-7890.\n"
            "Meeting on 05/20/2021 at 530 Broadway, San Diego, CA 92101."
        )
        self.targets = ['names', 'phones', 'dates', 'addresses']
        self.stats = {'names': 0, 'dates': 0, 'phones': 0, 'addresses': 0, 'concepts': 0}
    
    def test_no_matches(self):
        text = "No sensitive information here."
        expected_spans = []
        result = redact_entities_regex(text, self.targets, self.stats)
        self.assertEqual(result, expected_spans)
        self.assertEqual(self.stats, {'names': 0, 'dates': 0, 'phones': 0, 'addresses': 0, 'concepts': 0})
    
    def test_overlapping_matches(self):
        text = "Johnathan Doe's phone is 1234567890 and he lives at 123 Main St."
        expected_spans = [
            (0, 13),    
            (56, 63),   
            (25, 35),
            (52, 64),  
        ]
        result = redact_entities_regex(text, self.targets, self.stats)
        self.assertEqual(result, expected_spans)
        self.assertEqual(self.stats, {'names': 2, 'dates': 0, 'phones': 1, 'addresses': 1, 'concepts': 0})

if __name__ == '__main__':
    unittest.main()
