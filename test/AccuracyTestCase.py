import sys
sys.path.insert(0, '../')
import os
import unittest
from train import get_accuracy
import numpy as np

class AccuracyTestCase(unittest.TestCase):
	
	
	def test_accuracy(self):

		target = 	np.array([	
							[1,2,3,0,0,0,0],
							[1,2,3,0,0,0,0],
							[1,2,3,0,0,0,0] 
						])
		
		logits	= 	np.array([	
							[[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]],
							[[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]],
							[[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]]
						])

		acc = get_accuracy(target, logits)
		self.assertEqual(acc, 1.)	

		

		target = np.array([ 63,64,282,292,45,73,105,37,323,83,282,236,205,73,189,46,1,0,0,0,0])
		hyp = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
		print(np.mean(np.equal(target, hyp)))
		#print(get_accuracy(target, logits))

if __name__ == '__main__':
	unittest.main()
