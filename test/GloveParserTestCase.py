import sys
sys.path.insert(0, '../')
import unittest
import numpy as np
from mimic_utility_classes import GloveParser

class GloveParserTestCase(unittest.TestCase):
    def setUp(self):
        self.parser = GloveParser("test_resources/test_glove.vocab")
        self.parser.parse()
        self.vocab = self.parser.get_vocabulary()
        self.weights = self.parser.get_weights()
        self.ndarray = self.parser.get_weights_as_ndarray()

    def test_a_glove_parser_requires_the_filename_containing_the_weigths(self):
        self.assertTrue(isinstance(self.parser, GloveParser))

    def test_given_a_parser_when_parse_then_parse_the_contents_in_the_file(self):
        self.parser.parse() 

    def test_given_a_parser_and_parse_then_get_vocab_returns_vocab(self):
        self.assertTrue(isinstance(self.vocab, dict))

    def test_given_a_vocab_then_the_exist_in_vocab(self):
        self.assertTrue('the' in self.vocab)
        self.assertTrue('.' in self.vocab)
        self.assertTrue('of' in self.vocab)
        self.assertTrue('and' in self.vocab)
        self.assertEqual(len(self.vocab), 15)

    def test_given_glove_parser_and_parse_when_get_weights_then_weights_of_words(self):
        self.assertTrue(isinstance(self.weights, dict))

    def test_when_get_weights_the_weigths_size_50(self):
        for weights in self.weights.values():
            self.assertTrue(isinstance(weights, list))
            self.assertTrue(len(weights), 50)

    def test_weigths_and_vocab_have_the_same_size(self):
        self.assertTrue(len(self.vocab), len(self.weights))
        
    def test_given_glove_parser_when_get_weights_as_ndarray_then_get_ndarray(self):
        self.assertTrue(isinstance(self.ndarray, np.ndarray))

    def test_ndarray_and_weights_must_have_equal_size_and_same_contents(self):
        self.assertEqual(len(self.weights), len(self.ndarray))
        for idx in range(len(self.weights)):
            res = np.array(self.weights[idx]) == self.ndarray[idx]
            self.assertTrue(res.all())

    def test_parser_should_return_embedding_size(self):
        self.assertEqual(self.parser.get_embedding_dimension(), 50)
    
    def test_parser_should_return_always_the_same_word2idx(self):
        self.assertEqual(self.vocab['the'], 0)
        self.assertEqual(self.vocab[','], 1)
        self.assertEqual(self.vocab['.'], 2)
        self.assertEqual(self.vocab['of'], 3)
        self.assertEqual(self.vocab['to'], 4)
        self.assertEqual(self.vocab['and'], 5)
        self.assertEqual(self.vocab['in'], 6)
        self.assertEqual(self.vocab['a'], 7)


if __name__ == "__main__":
    unittest.main()
