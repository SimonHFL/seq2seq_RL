import sys 
sys.path.insert(0, '../')
from mimic_utility_classes import GloveParser
from mimic_utility_classes import MimicDataProcessor 
import unittest
import numpy as np
import tensorflow as tf

class EvalGraphTestCase(unittest.TestCase):
    
    def setUp(self):
        glove = "test_resources/glove/glove.6B.50d.txt"
        parser = GloveParser(glove)
        parser.parse()
        self.glove_weights = parser.get_weights_as_ndarray()
        self.proc = MimicDataProcessor(parser.get_vocabulary())
        self.max_length = self.proc.max_length
        self.proc.read_analogies("test_resources/questions-words.txt")
        self.a, self.b, self.c, self.d = self.proc.analogy_questions_with_answers
        print(self.a.shape)

    @unittest.skip("")
    def test_get_analogies(self):        
        raw_dimensions = 64 * self.max_length
        with tf.Graph().as_default():
            ngram_embeddings = tf.Variable(
                    tf.random_uniform([len(self.proc.get_ngram_vocab()), 64], -1, 1))
            glove_embeddings = tf.placeholder(
                    shape=self.glove_weights.shape, dtype=tf.float32)

            a_analogy = tf.placeholder(
                    dtype=tf.int32, shape=self.a.shape)

            embed_a = tf.nn.embedding_lookup(ngram_embeddings, a_analogy)
            labels = tf.nn.embedding_lookup(glove_embeddings, self.d)

            reshaped_a = tf.reshape(embed_a, shape=[self.a.shape[0], -1])

            init = tf.global_variables_initializer()
            with tf.Session() as session:
                init.run()
                a = session.run([labels], feed_dict={glove_embeddings: self.glove_weights})
                print(a)


            
if __name__ == "__main__":
    unittest.main()
