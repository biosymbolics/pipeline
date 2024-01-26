import unittest

from utils.list import merge_nested


class TestListUtils(unittest.TestCase):
    def test_merge(self):
        obj1 = {"a": {"a1": [1]}, "b": {"a1": [2]}}
        obj2 = {"a": {"a1": [3]}, "b": {"a1": [4]}}

        expected = {"a": {"a1": [1, 3]}, "b": {"a1": [2, 4]}}
        result = merge_nested(obj1, obj2)
        print(result)

        self.assertEqual(result, expected)
