import sys
sys.path.insert(0, '../')
from mimic_utility_classes import NgramTextProcessor
import unittest
import numpy as np

class NgramTextProcessorTestCase(unittest.TestCase):

    def setUp(self):
        self.data = "This is a sentence.\nThis is another"
        self.proc = NgramTextProcessor(self.data, ngram_order=1, pad_word=True, pad_to_max=True)
        self.n2i, self.i2n = self.proc.ngram2idx, self.proc.idx2ngram
        self.data = "This is a text.\nThis is another."

    def test_NgramTextProcessor_accepts_data_ngram_order(self):
        self.assertTrue(isinstance(self.proc, NgramTextProcessor))

    def test_given_proc_when_get_vocab_then_return_ngram2idx_idx2ngram(self):
        self.assertTrue(isinstance(self.n2i, dict))
        self.assertTrue(isinstance(self.i2n, dict))

    def test_given_proc_when_get_vocab_then_returned_appropriate_vocab(self):
        self.assertIn('i', self.n2i)
        self.assertIn('a', self.n2i)
        self.assertIn('t', self.n2i)
        self.assertIn('t', self.n2i)
        self.assertNotIn('ta', self.n2i)
        self.assertNotIn('w', self.n2i)

    def test_given_proc_and_text_when_text_to_ngram_indices_then_get_ngram_indices_as_ndarray(self):
        ngram_indices = self.proc.text_to_ngram_indices("make each word a list of ngram indices")
        self.assertTrue(isinstance(ngram_indices, np.ndarray))

    def test_given_proc_and_text_then_return_accurate_ngram_indices(self):
        ngram_indices = self.proc.text_to_ngram_indices(
                "make each word a list.\nfor each sentence")
        self.assertEqual(len(ngram_indices), 2)
        self.assertEqual(len(ngram_indices[0]), 5)
        self.assertEqual(len(ngram_indices[1]), 3)

    def test_given_proc__text_and_pad_to_max_then_all_lenght_of_word_list_equal_to_max(self):
        ngram_indices = self.proc.text_to_ngram_indices(
                "make each word a  list.\nfor each sentence")
        for word in ngram_indices[0]:
            self.assertEqual(len(self.proc._make_word_ngram("sentence")), len(word))
        print(self.proc.text_to_ngram_indices("another"))

        
if __name__ == "__main__":
    unittest.main()
