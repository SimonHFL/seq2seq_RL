import sys
sys.path.insert(0, '../')
from mimic_utility_classes import GloveParser
import unittest

class IntegrationGloveTestCase(unittest.TestCase):
    def setUp(self):
	    self.parser = GloveParser("test_resources/glove.6B.300d.txt")
	
    @unittest.skip("Take time")
    def test_integration(self):
        self.parser.parse()
        print(len(self.parser.get_vocabulary()))

if __name__ == "__main__":
    unittest.main()
