import sys
sys.path.insert(0, '../')
from mimic_utility_classes import DataPreprocessor
import numpy as np
import unittest 

class DataPreprocessorTestCase(unittest.TestCase):
    
    def setUp(self):
        self.data = "Test Data here is an a example"
        self.processor = DataPreprocessor(self.data, batch_size=8, ngram=2, pad_word=True,
                pad_to_max=True, lower_word=True, window=1)
        self.ngram2idx, self.idx2ngram, self.word2idx, self.idx2word = self.processor._create_vocab()

    def test_a_data_preprocessor_accept_data(self):
        processor = DataPreprocessor(self.data)
        self.assertTrue(isinstance(processor, DataPreprocessor))
    
    def test_a_processor_can_accept_batch_size_window_ngram_order_pad_word_size_word_lower_word(self):

        self.assertEqual(self.processor.batch_size, 8)
        self.assertEqual(self.processor.ngram, 2)
        self.assertTrue(self.processor.pad_word)
        self.assertEqual(self.processor.window, 1)

    def test_processor_creates_ngram2idx_idx2ngram_word2idx_idx2word_vocab(self):
        self.assertTrue(isinstance(self.ngram2idx, dict))
        self.assertTrue(isinstance(self.idx2ngram, dict))
        self.assertTrue(isinstance(self.word2idx, dict))
        self.assertTrue(isinstance(self.idx2word, dict))

    def test_processor_can_accept_to_either_load_or_create_vocab(self):
        processor = DataPreprocessor(self.data, batch_size=8, ngram=2, pad_word=True,
                pad_to_max=True, lower_word=True, window=1, load_data=True)
        self.assertTrue(isinstance(processor, DataPreprocessor))

    def test_processor_can_accept_to_the_dist_to_load_the_vocab(self):
        processor = DataPreprocessor(self.data, batch_size=8, ngram=2, pad_word=True,
                pad_to_max=True, lower_word=True, window=1, load_data=True)
        self.assertTrue(isinstance(processor, DataPreprocessor))


    def test_processor_creates_valid_vocab(self):
        self.assertIn('es', self.ngram2idx)
        self.assertIn('here', self.word2idx)
        self.assertNotIn('Te', self.ngram2idx)
        self.assertNotIn('exmples', self.ngram2idx)
        self.assertNotIn('Test', self.word2idx)
        self.assertIn('$t', self.ngram2idx)
        self.assertIn('n$', self.ngram2idx)

    def test_processor_vocabulary_contains_unk_pad_tokens(self):
        self.assertTrue('<UNK>' in self.ngram2idx)
        self.assertTrue('<PAD>' in self.ngram2idx)

    def test_processor_should_generate_batches_with_size_batch_size(self):
        inps, labs = self.processor.generate_batch()
        self.assertEqual(len(inps), 8)
        self.assertEqual(len(labs), 8)

    def test_the_batches_must_be_valid_indices(self):
        inps, labs = self.processor.generate_batch()
        for ngramidx in inps[0]:
            self.assertTrue(ngramidx in self.idx2ngram)
        widx = labs[0, 0]
        self.assertTrue(widx in self.idx2word)

    def test_the_batches_returned_should_be_numpy_arrays(self):
        inps, labs = self.processor.generate_batch()
        self.assertTrue(isinstance(inps, np.ndarray))
        self.assertTrue(isinstance(labs, np.ndarray))

    def test_max_length_should_be_equal_to_max_ngram_length_in_ngram_vocab(self):
        ml = self.processor.max_length
        mle = len(self.processor._make_word_ngram("example"))
        self.assertEqual(mle, ml)

    def test_batch_dimensions(self):
        inps, labs = self.processor.generate_batch()
        self.assertEqual(inps.shape, (self.processor.batch_size, self.processor.max_length))

    def test_make_word_ngram(self):
        processor = DataPreprocessor(self.data, ngram=4, pad_word=True)
        l = processor._make_word_ngram("a")
        self.assertEqual(len(l), 1)



if __name__ == "__main__":
    unittest.main()
