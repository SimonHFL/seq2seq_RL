import sys
sys.path.insert(0, '../')
import unittest
from mimic_utility_classes import DataPreprocessor

def make_word(ngrams, idx2ngram):
    return ''.join([idx2ngram[ng] for ng in ngrams])

class IntegrationDataSetTestCase(unittest.TestCase):

    def setUp(self):
        with open("test_resources/test_data.txt", 'r') as f:
            self.test_data = f.read()
        self.processor = DataPreprocessor(self.test_data, ngram=2, pad_word=True, window=4)

    @unittest.skip("")
    def test_generate_batches(self):
        for _ in range(8):
            inps, labs = self.processor.generate_batch()
            self.assertEqual(len(inps), 16)
            self.assertEqual(len(labs), 16)
    
    @unittest.skip("")
    def test_data_generation(self):
        for _ in range(2):
            inps, labs = self.processor.generate_batch()

if __name__ == "__main__":
    unittest.main()
