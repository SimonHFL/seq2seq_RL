import sys, os
sys.path.insert(0, '../')
import unittest

from data_preprocessing import get_data
from data_preprocessing import get_test
from helper import create_vocabulary


class DataProcessingTestCase(unittest.TestCase):

    def setUp(self):
        self.get_data = lambda x, y: get_data(
                train_source_path = 'test_resources/test_small_vocab.txt',
                train_target_path = 'test_resources/test_small_vocab.txt',
                valid_source_path = 'test_resources/test_revised_conll13.input',
                valid_target_path = 'test_resources/test_revised_conll13.input',
                enc_ngram_order_tokenization = x,
                dec_ngram_order_tokenization = y)
        self.text = "this is one text."

    def tearDown(self):
        for f in os.listdir():
            if f.endswith('.p'):
                os.remove(f)

    def test_get_data_should_accept_tokenization_policy_both_for_encoder_and_decoder(self):
        data  = self.get_data(None, None)
        self.assertEqual(4, len(data))

    def test_get_data_given_enc_dec_tokenization_word_when_get_data_then_word_vocab(self):
        _, _, vocab, _ = self.get_data(None, None)
        source_word2idx, target_word2idx = vocab
        self.assertIn("sometimes", source_word2idx)
        self.assertIn("favorite", target_word2idx)

    def test_get_vocab_accepts_tokenization_policy(self):
        word2idx, idx2word = create_vocabulary(self.text, ngram_order=None)
        self.assertTrue(isinstance(word2idx, dict))
        self.assertIn("this", word2idx)

    def test_get_vocab_when_ngram_order_1_then_relevant_vocabulary_is_returned(self):
        char2idx, idx2char = create_vocabulary(self.text, ngram_order=1)
        self.assertIn('t', char2idx)
        self.assertIn('o', char2idx)
        self.assertIn('n', char2idx)

    def test_get_data_given_ngram_order_for_both_target_and_source_then_get_appropriate_vocabulary(self):
        _, _, vocab, _ = self.get_data(1, 2)
        source, target = vocab
        self.assertIn('t',  source)
        self.assertIn('th', target)

    def test_loading_the_test_set(self):
        test_input, test_target = get_test(
            test_source_file='test_resources/test_small_vocab.txt',
            test_target_file='test_resources/test_small_vocab.txt',
            pickle='test_resources/preprocess_word.p')
        for sentence in test_input:
            for word in sentence:
                self.assertTrue(isinstance(word, int))

if __name__ == "__main__":
    unittest.main()
