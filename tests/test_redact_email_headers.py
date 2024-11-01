import unittest
from unittest.mock import patch, Mock
from redactor import redact_email_headers

class TestRedactEmailHeaders(unittest.TestCase):
    def setUp(self):
        self.text = (
            "From: John Doe <john.doe@example.com>\n"
            "To: Jane Smith <jane.smith@example.com>\n"
            "Subject: Meeting Schedule\n"
            "Best regards,\n"
            "John"
        )
        self.targets = ['names']
        self.stats = {'names': 0, 'dates': 0, 'phones': 0, 'addresses': 0, 'concepts': 0}
    
    def test_redact_names_in_headers(self):
        expected_spans = [
            (6, 14),
            (16, 20), 
            (21, 24), 
            (42, 52), 
            (54, 58),   
            (59, 64)    
        ]
        result = redact_email_headers(self.text, self.targets, self.stats)
        self.assertEqual(result, expected_spans)
        self.assertEqual(self.stats, {'names': 6, 'dates': 0, 'phones': 0, 'addresses': 0, 'concepts': 0})
    
if __name__ == '__main__':
    unittest.main()
