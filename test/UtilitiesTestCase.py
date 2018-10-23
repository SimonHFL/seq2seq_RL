import sys
sys.path.insert(0, '../')
import unittest
import os
from mimic_utility_defs import reduce_dimensions_and_draw
from mimic_utility_defs import dump
from mimic_utility_defs import load
from mimic_utility_defs import len_of_longest_word

import numpy as np

class UtilitiesTestCase(unittest.TestCase):
    def setUp(self):
        self.dict = {'word1' : np.ndarray(shape=[300], dtype=np.int32), 
                'word2' : np.ndarray(shape=[300], dtype=np.int32),
                'word3' : np.ndarray(shape=[300], dtype=np.int32)}


    def test_accepts_a_label2vector_dict_to_reduce_and_optional_filename(self):
        reduce_dimensions_and_draw(self.dict)

    def test_dump_takes_an_object_to_store_and_filename(self):
        dump(("aaa", "bbb"), 'test.p')
        self.assertTrue(os.path.exists('test.p'))

    def test_load_takes_a_file_and_returns_the_object(self):
        self.assertEqual(load('test.p'), ("aaa", "bbb"))
        os.remove('test.p')

    def test_len_of_longest_word(self):
        data = ["One sentence", "Another one"]
        self.assertEqual(8, len_of_longest_word(data))

if __name__ == "__main__":
    unittest.main()
