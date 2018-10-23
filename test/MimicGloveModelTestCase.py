import sys
sys.path.insert(0, '../')
import math
import unittest
import numpy as np

from mimic_utility_classes import GloveParser
from mimic_utility_classes import MimicDataProcessor
from six.moves import xrange

import tensorflow as tf

class MimicGloveModelTestCase(unittest.TestCase):
    
    def setUp(self):
        glove_file = 'test_resources/glove/glove.6B.50d.txt'
        glove_file1 = 'test_resources/test_glove.vocab'
        parser = GloveParser(glove_file)
        parser.parse()
        self.ngram=3
        self.batch_size=128
        self.glove_embed_size = parser.get_embedding_dimension()
        self.glove_vocab = parser.get_vocabulary()
        self.word_vocab_size = len(self.glove_vocab)
        self.glove_weights = parser.get_weights_as_ndarray()
        self.mimic_proc = MimicDataProcessor(self.glove_vocab, ngram_order=self.ngram,
                batch_size=self.batch_size)
        self.max_length = self.mimic_proc.max_length
        self.ngram_embed_size = 64
        self.dimension = self.ngram_embed_size * self.max_length
        self.ngram_vocab_size = len(self.mimic_proc.get_ngram_vocab())
        self.hidden1_units = 32 * self.max_length

    @unittest.skip("")
    def test_word_vector_prediction(self):
        graph = tf.Graph()
        with graph.as_default():
            train_inputs = tf.placeholder(
                    tf.int32, shape=[self.batch_size, self.max_length])
            train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            
            with tf.device('/cpu:0'):
                embeddings = tf.Variable(
                    tf.random_uniform(
                        [self.ngram_vocab_size, self.ngram_embed_size], -1, 1))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)
                res_inputs = tf.reshape(embed, shape=[self.batch_size, -1])

                w1 = tf.Variable(
                    tf.truncated_normal([self.dimension, self.hidden1_units], 
                        stddev=1.0/math.sqrt(float(self.dimension))))

                b1 = tf.Variable(
                    tf.zeros([self.hidden1_units]))
        
                hidden1 = tf.nn.relu(tf.matmul(res_inputs, w1) + b1)

                w2 = tf.Variable(
                    tf.truncated_normal([self.hidden1_units, self.glove_embed_size], 
                        stddev=1.0/math.sqrt(float(self.glove_embed_size))))

                b2 = tf.Variable(tf.zeros([self.glove_embed_size]))

                res_logits = tf.matmul(hidden1, w2) + b2
                shape = tf.shape(res_logits)

            init = tf.global_variables_initializer()
        num_steps = 1

        with tf.Session(graph=graph) as sess:
            init.run()
            for step in xrange(num_steps):
                batch_inputs, batch_labels = self.mimic_proc.generate_batches()
                feed_dict = {train_inputs: batch_inputs}
                rs_sh = sess.run([shape], feed_dict=feed_dict)
                res = rs_sh == np.array([128, 50], dtype=np.int32)
                self.assertTrue(res.all())

    @unittest.skip("")
    def test_load_gloves_in_constant_tensor(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/cpu:0'):
                const = tf.placeholder(tf.float32, shape=self.glove_weights.shape)
                s = tf.shape(const)

        with tf.Session(graph=graph) as sess:
            glove = sess.run([s], feed_dict = {const: self.glove_weights})
            res = glove == np.array([400000, 50], dtype=np.int32)
            self.assertTrue(res.all())

    @unittest.skip("")
    def test_euclidean_distance_loss(self):
        dummy_labels = np.array(list(range(self.batch_size)), dtype=np.int32).reshape(self.batch_size, 1)
        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/cpu:0'):
                dummy_prediction = tf.Variable(
                        tf.truncated_normal([self.batch_size, self.glove_embed_size], 
                            stddev=1.0/math.sqrt(float(self.glove_embed_size))))
                glove = tf.placeholder(
                        tf.float32, shape=[self.word_vocab_size, self.glove_embed_size])
                labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

                target = tf.nn.embedding_lookup(glove, labels)
                loss = tf.reduce_mean(tf.squared_difference(dummy_prediction, target))
                optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
            init = tf.global_variables_initializer()
        with tf.Session(graph=graph) as sess:
            init.run()
            for _ in range(10):
                feed_dict = {glove: self.glove_weights, labels: dummy_labels}
                #lookups = sess.run([target], feed_dict=feed_dict)
                #print(lookups)
                #print(sess.run([sh], feed_dict=feed_dict))
                _, lo = sess.run([optimizer ,loss], feed_dict=feed_dict)
                print(lo)
if __name__ == "__main__":
    unittest.main()
