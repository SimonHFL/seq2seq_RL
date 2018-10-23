import sys
sys.path.insert(0, '../')
import unittest
from mimic_utility_classes import _Node
from mimic_utility_classes import LinkedBatcher


class LinkedBatcherTestCase(unittest.TestCase):

    def setUp(self):
        self.batch_size = 4
        self.linked = LinkedBatcher(list(range(19)), batch_size=self.batch_size)
        self.head = self.linked._create_linked_batch_and_get_head(list(range(19)))
    
    def test_a_linked_batcher_accepts_list_and_batch_size(self):
        self.assertTrue(isinstance(self.linked, LinkedBatcher))

    def test_given_batch_size_greater_than_list_size_then_raise_ValueError(self):
        with self.assertRaises(ValueError):
            LinkedBatcher([1], batch_size=4)

    def test_given_linked_batcher_when_get_batches_then_get_list_equal_to_batch_size(self):
        batch = self.linked.get_batch()
        self.assertTrue(isinstance(batch, list))
        self.assertTrue(len(batch), self.batch_size)

    def test_given_linked_batcher_when_get_batch_multiple_times_then_different_batches(self):
        batch1 = self.linked.get_batch()
        batch2 = self.linked.get_batch()
        self.assertNotEqual(batch1, batch2)
    
    def test__create_linked_list_and_get_head(self):
        self.assertTrue(isinstance(self.head, _Node))

    def test_when_create_linked_and_get_head(self):
        self.assertEqual(0, self.head.value)

    @unittest.skip("Printing only")
    def test_integration(self):
        for _ in range(20):
            print(self.linked.get_batch())

if __name__ == "__main__":
    unittest.main()
