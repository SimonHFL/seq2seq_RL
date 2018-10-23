from __future__ import division
from __future__ import absolute_import
import sys
sys.path.insert(0, '../')
import unittest
import tensorflow as tf
import numpy as np
import math
from six.moves import xrange
from mimic_utility_classes import DataPreprocessor

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    import matplotlib.pyplot as plt
    assert low_dim_embs.shape[0] >= len(labels)
    plt.figure(figsize=(18,18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x,y)
        plt.annotate(label,
                xy=(x,y),
                xytext=(5,2),
                textcoords='offset points',
                ha='right',
                va='bottom')
    plt.savefig(filename)

def make_png(embeddings, plot_only, idx2ngram, filename):
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        low_dim_embs = tsne.fit_transform(embeddings[:plot_only, :])
        labels = [idx2ngram[i] for i in xrange(plot_only)]
        plot_with_labels(low_dim_embs, labels, filename)
    except ImportError:
        print('You know')



class LookupTestCase(unittest.TestCase):

    def setUp(self):
        self.batch_size = 8 
        with open('test_resources/test_data.txt', 'r') as f:
            self.test_data=f.read()
        self.processor = DataPreprocessor(self.test_data, batch_size=self.batch_size, ngram=2, pad_word=True,
                pad_to_max=True, verbose=True, window=4)

        self.max_length = self.processor.max_length
        self.ngram_size = len(self.processor.ngram2idx)
        self.word_size = len(self.processor.word2idx)
        self.embedding_size = 64
        print('Max ngram length: ', self.max_length, '\nWord voca: ', self.word_size,'\nNgram size: ', self.ngram_size)

    @unittest.skip("Takes time")
    def test_lookup_nram_emmbedings(self):
        valid_examples = 100
        valid_size = 16
        valid_window = 100
        valid_examples = np.random.choice(valid_window, valid_size, replace=False)
        xdimension = self.embedding_size * self.max_length
        graph = tf.Graph()
        with graph.as_default():
            train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_length])
            train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
            with tf.device('/cpu:0'):
                #Construct ngram embeddings
                embeddings = tf.Variable(
                        tf.random_uniform([self.ngram_size, self.embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)
                res = tf.reshape(embed, shape=[self.batch_size, -1])

                nce_weights = tf.Variable(
                        tf.truncated_normal([self.word_size, xdimension],
                            stddev=1.0/math.sqrt(xdimension)))
                nce_biases = tf.Variable(tf.zeros([self.word_size]))
                
            loss = tf.reduce_mean(
                    tf.nn.nce_loss(
                    weights=nce_weights,
                    biases=nce_biases,
                    labels=train_labels,
                    inputs=res,
                    num_sampled=7,
                    num_classes=self.word_size))

            optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings/norm

            #valid_embeddings=tf.nn.embedding_lookup(
            #normalized_embeddings, valid_dataset)
            #similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
            init = tf.global_variables_initializer()

        num_steps = 1
        
        with tf.Session(graph=graph) as session:
            init.run()
            print('Initialized')
            average_loss = 0
            make_png(embeddings.eval(), len(self.processor.ngram2idx), self.processor.idx2ngram, "before.png")
            for step in xrange(num_steps):
                batch_inputs, batch_labels = self.processor.generate_batch()
                feed_dict = {train_inputs:batch_inputs, train_labels:batch_labels}
                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss +=loss_val

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000 
                    print('Average_loss at step', step, ':', average_loss)
                    average_loss = 0

            final_embeddings = normalized_embeddings.eval()
            make_png(final_embeddings, len(self.processor.ngram2idx), self.processor.idx2ngram, "after.png")

if __name__ == "__main__":
    unittest.main()

