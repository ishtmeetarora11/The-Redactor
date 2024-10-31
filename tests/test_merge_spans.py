import unittest
from redact import merge_overlapping_spans

class TestMergeOverlappingSpans(unittest.TestCase):
    def test_no_overlaps(self):
        spans = [(0, 5), (10, 15), (20, 25)]
        expected = [(0, 5), (10, 15), (20, 25)]
        result = merge_overlapping_spans(spans)
        self.assertEqual(result, expected)
    
    def test_with_overlaps(self):
        spans = [(0, 10), (5, 15), (20, 25)]
        expected = [(0, 15), (20, 25)]
        result = merge_overlapping_spans(spans)
        self.assertEqual(result, expected)
    
    def test_adjacent_spans(self):
        spans = [(0, 5), (5, 10), (15, 20)]
        expected = [(0, 10), (15, 20)]
        result = merge_overlapping_spans(spans)
        self.assertEqual(result, expected)
    
    def test_nested_spans(self):
        spans = [(0, 20), (5, 10), (15, 25)]
        expected = [(0, 25)]
        result = merge_overlapping_spans(spans)
        self.assertEqual(result, expected)
    
    def test_empty_spans(self):
        spans = []
        expected = []
        result = merge_overlapping_spans(spans)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()