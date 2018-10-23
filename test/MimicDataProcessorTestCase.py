import sys
sys.path.insert(0, '../')
import unittest
import numpy as np
from mimic_utility_classes import MimicDataProcessor
from mimic_utility_classes import GloveParser

class MimicDataProcessorTestCase(unittest.TestCase):
    def setUp(self):
        parser = GloveParser('test_resources/test_glove.vocab')
        parser.parse()
        self.eval_data = 'test_resources/questions-words.txt'
        self.ngram_order = 2
        self.batch_size = 8
        self.vocab = parser.get_vocabulary()
        self.proc = MimicDataProcessor(self.vocab, ngram_order=2, pad_word=True, batch_size=self.batch_size)
        self.inputs, self.labels = self.proc.generate_batches()

    
    def test_mimic_processor_should_accept_a_vocab_and_an_optional_ngram_order_pad_word_option_and_number_of_batch_size(self):
        self.assertTrue(isinstance(self.proc, MimicDataProcessor))

    def test_mimic_processor_accepts_flag_either_to_load_data_or_not_and_dest(self):
        proc = MimicDataProcessor(self.vocab, ngram_order=2, pad_word=True, batch_size=self.batch_size, load_data=False, dest='mimic_data_test.p')
        self.assertTrue(isinstance(proc, MimicDataProcessor))
    
    def test_given_processor_when_generate_batch_then_get_batches_equal_to_batch_size(self):
        self.assertTrue(len(self.inputs), self.batch_size)
        self.assertTrue(len(self.labels), self.batch_size)

    def test_processor_should_return_ngram_vocab(self):
        self.assertTrue(isinstance(self.proc.get_ngram_vocab(), dict))
        self.assertTrue(isinstance(self.proc.get_index_to_ngram(), dict))

    def test_should_return_the_word_maxlength(self):
        self.assertTrue(isinstance(self.proc.max_length, int))

    def test_given_a_batch_all_the_indidices_of_ngram_should_be_valid(self):
        for inp in self.inputs:
            for idx in inp:
                self.assertTrue(idx in self.proc.dataProcessor.idx2ngram)

    def test_given_analogy_file_when_read_analogies_then_ndarray_with_analogies(self):
        self.proc.read_analogies(self.eval_data)
        analogies = self.proc.analogy_questions_with_answers
        self.assertTrue(isinstance(analogies, tuple))
        self.assertTrue(len(analogies), 4)

    def test_given_read_analogies_then_when_n_analogies_then_number_of_analogies(self):
        self.proc.read_analogies(self.eval_data)
        self.assertEqual(self.proc.n_analogies, 0)

    def test_given_words_when_get_ngram_indices_for_words_then_return_an_np_array(self):
        words = ["the", "that"]
        indices = self.proc.get_ngram_indices_from(words)
        self.assertTrue(isinstance(indices, np.ndarray))
        self.assertEqual(len(indices), len(words))

    def test_given_words_when_get_ngram_indices_then_get_a_list_of_ngram_indices_for_each_word_in_words(self):
        words = ["the", "thth"]
        lists_of_indices = self.proc.get_ngram_indices_from(words)
        for list_of_indices in lists_of_indices:
            self.assertEqual(len(list_of_indices), self.proc.dataProcessor.max_length)
            for idx in list_of_indices:
                self.assertIn(idx, self.proc.dataProcessor.idx2ngram)

if __name__ == "__main__":
    unittest.main()
