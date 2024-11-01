# tests/test_process_file.py

import unittest
from unittest.mock import patch, mock_open
import os
import tempfile
from redactor import process_file

class TestProcessFile(unittest.TestCase):
    @patch('redactor.redact_email_headers')
    @patch('redactor.redact_entities_spacy')
    @patch('redactor.redact_entities_hf')
    @patch('redactor.redact_entities_regex')
    @patch('redactor.merge_overlapping_spans')
    @patch('builtins.open', new_callable=mock_open, read_data="Original content")
    def test_process_file_success(
        self, mock_file, mock_merge, mock_regex, mock_hf, mock_spacy, mock_email_headers
    ):
        # Setup mock return values within the text length (16 characters)
        mock_email_headers.return_value = [(0, 5)]      # Redact "Orig"
        mock_spacy.return_value = [(6, 10)]             # Redact "inal"
        mock_hf.return_value = [(11, 15)]               # Redact " cont"
        mock_regex.return_value = [(16, 20)]            # Redact beyond text length
        mock_merge.return_value = [(0, 16)]             # Merge spans up to 16

        # Setup arguments
        args = unittest.mock.Mock()
        args.names = True
        args.dates = False
        args.phones = True
        args.address = True
        args.concept = None
        args.output = tempfile.gettempdir()

        # Initialize stats with all keys
        stats = {
            'names': 1,
            'dates': 0,
            'phones': 1,
            'addresses': 1,
            'concepts': 0,
        }

        # Create a temporary input file
        input_file = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8')
        input_file.write("Original content")
        input_file.close()

        # Call process_file
        process_file(input_file.name, args, stats)

        # Check that the input file was read
        mock_file.assert_any_call(input_file.name, 'r', encoding='utf-8')

        # Check that the output file was written
        censored_file = os.path.join(args.output, os.path.basename(input_file.name) + ".censored")
        mock_file.assert_any_call(censored_file, 'w', encoding='utf-8')

        # Ensure that write was called with correctly redacted text
        handle = mock_file()
        expected_redacted_text = '█' * 16  # All characters replaced with '█'
        handle.write.assert_called_once_with(expected_redacted_text)

        # Check stats
        expected_stats = {
            'names': 1,       # From redact_email_headers
            'dates': 0,       # Dates not targeted
            'phones': 1,      # From redact_entities_regex
            'addresses': 1,   # From redact_entities_spacy
            'concepts': 0     # No concepts
        }
        self.assertEqual(stats, expected_stats)

    @patch('redactor.sys.stderr')
    @patch('builtins.open', side_effect=Exception("File read error"))
    def test_process_file_read_error(self, mock_open_file, mock_stderr):
        # Setup arguments
        args = unittest.mock.Mock()
        args.names = True
        args.dates = False
        args.phones = True
        args.address = True
        args.concept = None
        args.output = tempfile.gettempdir()

        # Initialize stats
        stats = {
            'names': 0,
            'dates': 0,
            'phones': 0,
            'addresses': 0,
            'concepts': 0,
        }

        # Call process_file with a nonexistent file
        process_file('nonexistent.txt', args, stats)

        # Check that an error message was written to stderr
        mock_stderr.write.assert_called_once_with("Error reading file nonexistent.txt: File read error\n")

if __name__ == '__main__':
    unittest.main()
