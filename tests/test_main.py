import unittest
import sys
from unittest.mock import patch, Mock
from redactor import main

class TestMainFunction(unittest.TestCase):
    @patch('redactor.glob.glob', return_value=['sample1.txt', 'sample2.txt'])
    @patch('redactor.process_file')
    @patch('redactor.write_stats')
    def test_main_success(self, mock_write_stats, mock_process_file, mock_glob):
        test_args = [
            'redactor.py',
            '--input', '*.txt',
            '--output', 'redacted_files',
            '--names',
            '--phones',
            '--address',
            '--concept', 'Confidential',
            '--stats', 'stdout'
        ]
        with patch.object(sys, 'argv', test_args):
            main()
            mock_glob.assert_called_with('*.txt')
            self.assertEqual(mock_process_file.call_count, 2)
            mock_write_stats.assert_called_once()
    
    @patch('redactor.glob.glob', return_value=[])
    @patch('redactor.sys.stderr')
    def test_main_no_files_matched(self, mock_stderr, mock_glob):
        test_args = [
            'redactor.py',
            '--input', '*.txt',
            '--output', 'redacted_files',
            '--names',
            '--phones',
            '--address',
            '--concept', 'Confidential',
            '--stats', 'stdout'
        ]
        with patch.object(sys, 'argv', test_args):
            main()
            mock_glob.assert_called_with('*.txt')
            mock_stderr.write.assert_called_with("No files matched the pattern: *.txt\n")
    
    @patch('redactor.glob.glob', return_value=['sample1.txt'])
    @patch('redactor.process_file')
    @patch('redactor.write_stats')
    def test_main_with_single_file(self, mock_write_stats, mock_process_file, mock_glob):
        test_args = [
            'redactor.py',
            '--input', '*.txt',
            '--output', 'redacted_files',
            '--names',
            '--phones',
            '--address',
            '--stats', 'stderr'
        ]
        with patch.object(sys, 'argv', test_args):
            main()
            mock_glob.assert_called_with('*.txt')
            mock_process_file.assert_called_once_with('sample1.txt', unittest.mock.ANY, unittest.mock.ANY)
            mock_write_stats.assert_called_once()

if __name__ == '__main__':
    unittest.main()
