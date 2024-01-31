import unittest

from utils.tensor import *


class TestUtils(unittest.TestCase):
    def test_array_to_tensor(self):
        test_conditions = [
            {
                "name": "test1",
                "data": [[1, 2, 3], [4, 5, 6]],
                "shape": (2, 3),
                "expected": torch.LongTensor([[1, 2, 3], [4, 5, 6]]),
            },
            {
                "name": "test1.5",
                "data": [[1.0, 2.3, 3.0], [4.0, 5.1, 6.2]],
                "shape": (2, 3),
                "expected": torch.FloatTensor([[1.0, 2.3, 3.0], [4.0, 5.1, 6.2]]),
            },
            {
                "name": "test2",
                "data": [[1, 2, 3], [4, 5]],
                "shape": (2, 3),
                "expected": torch.LongTensor([[1, 2, 3], [0, 4, 5]]),
            },
            {
                "name": "test3",
                "data": [torch.IntTensor([1, 2, 3]), torch.IntTensor([4, 5])],
                "shape": (2, 3),
                "expected": torch.IntTensor([[1, 2, 3], [0, 4, 5]]),
            },
            {
                "name": "test4",
                "data": [torch.IntTensor([1, 2, 3])],
                "shape": (1, 3),
                "expected": torch.IntTensor([1, 2, 3]),
            },
            {
                "name": "test5",
                "data": np.array([[1, 2, 3], [4, 5]], dtype=object),
                "shape": (2, 3),
                "expected": torch.LongTensor([[1, 2, 3], [0, 4, 5]]),
            },
            {
                "name": "test6",
                "data": [np.array([1, 2, 3], dtype=int), np.array([4, 5], dtype=int)],
                "shape": (2, 3),
                "expected": torch.LongTensor([[1, 2, 3], [0, 4, 5]]),
            },
            {
                "name": "test7",
                "data": [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
                "shape": (2, 2, 3),
                "expected": torch.LongTensor(
                    [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
                ),
            },
            {
                "name": "test8",
                "data": [[[1, 2, 3]], [[7, 8, 9], [10, 11, 12]]],
                "shape": (2, 2, 3),
                "expected": torch.LongTensor(
                    [[[0, 0, 0], [1, 2, 3]], [[7, 8, 9], [10, 11, 12]]]
                ),
            },
            {
                "name": "test9",
                "data": [np.array([1, 2, 3]), np.array([[7, 8, 9], [10, 11, 12]])],
                "shape": (2, 2, 3),
                "expected": torch.LongTensor(
                    [[[0, 0, 0], [1, 2, 3]], [[7, 8, 9], [10, 11, 12]]]
                ),
            },
            {
                "name": "test10",
                "data": [244, 389, 977],
                "shape": (4,),
                # TODO
                "expected": torch.LongTensor([0, 244, 389, 977]),
            },
            {
                "name": "Test 11 - Zero sized child tensors",
                "data": [torch.LongTensor([1, 2, 3]), torch.LongTensor()],
                "shape": (1, 3),
                "expected": torch.LongTensor([1, 2, 3]),
            },
            {
                "name": "Test 12 - empty",
                "data": [torch.Tensor()],
                "shape": (1, 3),
                "expected": torch.Tensor([0, 0, 0]),
            },
        ]

        for test in test_conditions:
            print("Starting test: ", test["name"])
            result = array_to_tensor(test["data"], test["shape"])
            is_equal = torch.all(torch.eq(test["expected"], result))
            if not is_equal:
                print(
                    "Expected for",
                    test["expected"],
                    "Actual:",
                    result,
                )
            self.assertTrue(is_equal)

            type_consistent = test["expected"].type() == result.type()
            if not type_consistent:
                print(
                    "Expected type:",
                    test["expected"].type(),
                    "Actual type:",
                    result.type(),
                )
            self.assertTrue(type_consistent)

        # self.assertTrue(False)
