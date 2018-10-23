import sys
sys.path.insert(0, '../')
import os
import unittest
from data_preprocessing import text_to_ids
from helper import create_vocabulary


class TextToIdsTestCase(unittest.TestCase):
	
	def setUp(self):
		self.target = "This is the vocabulary"
		self.source= "This i the vocab"
		self.word2idx, _ = create_vocabulary(self.target)
		self.input, self.output = text_to_ids(
			self.source, self.target, self.word2idx, self.word2idx)

		
	def test_text_to_ids(self):
		expected = self.target.split()
		self.assertEqual(len(expected), len(self.input[0]))	
	
	def test_the_idx_must_be_correct(self):
		unk_id = self.word2idx['<UNK>']
		self.assertEqual(unk_id, self.input[0][1])
		self.assertEqual(unk_id, self.input[0][3])
		
		

if __name__ == '__main__':
	unittest.main()
