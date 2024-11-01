# tests/test_write_stats.py

import unittest
from unittest.mock import patch, mock_open
from redactor import write_stats

class TestWriteStats(unittest.TestCase):
    def setUp(self):
        self.stats = {
            'names': 5,
            'dates': 3,
            'phones': 2,
            'addresses': 4,
            'concepts': 1
        }
    
    @patch('builtins.open', new_callable=mock_open)
    def test_write_stats_to_file(self, mock_file):
        destination = 'stats.txt'
        write_stats(self.stats, destination)
        mock_file.assert_called_with(destination, 'w', encoding='utf-8')
        handle = mock_file()
        expected_content = (
            "Names redacted: 5\n"
            "Dates redacted: 3\n"
            "Phone numbers redacted: 2\n"
            "Addresses redacted: 4\n"
            "Concepts redacted: 1\n"
        )
        handle.write.assert_called_once_with(expected_content)
    
    def test_write_stats_to_stdout(self):
        with patch('sys.stdout') as mock_stdout:
            write_stats(self.stats, 'stdout')
            expected_content = (
                "Names redacted: 5\n"
                "Dates redacted: 3\n"
                "Phone numbers redacted: 2\n"
                "Addresses redacted: 4\n"
                "Concepts redacted: 1\n"
            )
            mock_stdout.write.assert_called_once_with(expected_content)
    
    def test_write_stats_to_stderr(self):
        with patch('sys.stderr') as mock_stderr:
            write_stats(self.stats, 'stderr')
            expected_content = (
                "Names redacted: 5\n"
                "Dates redacted: 3\n"
                "Phone numbers redacted: 2\n"
                "Addresses redacted: 4\n"
                "Concepts redacted: 1\n"
            )
            mock_stderr.write.assert_called_once_with(expected_content)
    
    @patch('builtins.open', side_effect=Exception("File write error"))
    @patch('sys.stderr')
    def test_write_stats_to_file_failure(self, mock_stderr, mock_file):
        destination = 'invalid/path/stats.txt'
        write_stats(self.stats, destination)
        mock_stderr.write.assert_called_once_with(
            f"Failed to write statistics to {destination}: File write error\n"
        )

if __name__ == '__main__':
    unittest.main()
