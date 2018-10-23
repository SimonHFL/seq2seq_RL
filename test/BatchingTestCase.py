import sys
sys.path.insert(0, '../')
import os
import unittest
from helper import batch_data
import numpy as np

class BatchingTestCase(unittest.TestCase):
	
	
	def test_batching(self):
		#TODO: assert padding, assert reverse sorted by source len

		source = 	np.array([	
							[1,0,0,0,0,0,0,0],
							[2,0,0,0,0,0,0],
							[3,0,0,0,0,0],
							[4,0,0,0,0],
							[5,0,0,0],
						])

		target = 	np.array([	
							[1,0,0,0,0,0,0,0],
							[2,0,0,0,0,0,0],
							[3,0,0,0,0,0],
							[4,0,0,0,0],
							[5,0,0,0],
						])
		
		batch_size = 2
		for source_batch, target_batch in batch_data(source, target, batch_size):
			print("a")
			print(source_batch)
			#print(target_batch)
			

if __name__ == '__main__':
	unittest.main()
