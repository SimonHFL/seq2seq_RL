#Standard Python
import argparse
import math
import sys
import os
import time

#Thesis modules
import mimic_utility_classes

#External dependencies
from six.moves import xrange
import tensorflow as tf

FLAGS = None
GLOVE_DIM = None
NGRAM_VOCAB_SIZE = None
NUMBER_OF_NGRAMS_IN_WORD = None 

class MimicGlove(object):

    def __init__(self, sess, options, glove_weights):
        self.session = sess
        self.options = options
        self.build_graph(glove_weights)

    def build_graph(self, glove_weights):
        """Build the full graph of the model"""

        self.glove_weights = tf.constant(glove_weights, dtype=tf.float32)
        self.train_inputs, self.train_labels = self.placeholder_inputs()

        logits = self.inference(self.train_inputs)
        self._logits = logits

        loss = self.loss(logits, self.train_labels) 
        tf.summary.scalar(FLAGS.loss, loss)
        self._loss = loss
        self._train = self.optimize(loss)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()


    def placeholder_inputs(self) -> "train_inputs train_labels and glove weights placeholder":

        train_inputs = tf.placeholder(
                tf.int32, shape=[None, 
                    self.options.number_of_ngrams_in_word], name='inputs_placeholder')
        train_labels = tf.placeholder(tf.int32, shape=[self.options.batch_size], name='labels_placeholder')

        return train_inputs, train_labels

    def inference(self, train_inputs: "Placeholder with ngram indices"):
        raw_word_dimension = self.options.ngram_embed_dim * self.options.number_of_ngrams_in_word
        with tf.name_scope('embedding'):
            init_width = 0.5/self.options.ngram_embed_dim
            ngram_embeddings = tf.Variable(
                    tf.random_uniform([self.options.ngram_vocab_size, self.options.ngram_embed_dim], -init_width, init_width),
                    name='ngram_embeddings')

            self.ngram_embeddings = ngram_embeddings
            embed = tf.nn.embedding_lookup(ngram_embeddings, train_inputs)

            reshaped = tf.reshape(embed, shape=[-1, raw_word_dimension])
        with tf.name_scope('hidden1'):
            self.w1 = tf.Variable(
                    tf.truncated_normal([raw_word_dimension, FLAGS.hidden1],
                        stddev=1.0/math.sqrt(float(raw_word_dimension))), 'weights')
            self.b1 = tf.Variable(tf.zeros([self.options.hidden1]), name='biases')

            hidden1 = tf.nn.relu(tf.matmul(reshaped, self.w1) + self.b1)

        with tf.name_scope('hidden2'):
            self.w2 = tf.Variable(
                tf.truncated_normal([self.options.hidden1, self.options.hidden2],
                    stddev=1.0/math.sqrt(float(FLAGS.hidden1))), 'weights')
            self.b2 = tf.Variable(tf.zeros([self.options.hidden2]), name = 'biases')

            hidden2 = tf.nn.relu(tf.matmul(hidden1, self.w2) + self.b2)

        with tf.name_scope('hidden3'):
            self.w3 = tf.Variable(
                tf.truncated_normal([self.options.hidden2, self.options.glove_word_emb_dim],
                    stddev=1.0/math.sqrt(float(self.options.hidden2))), 'weights')
            self.b3 =  tf.Variable(tf.zeros([self.options.glove_word_emb_dim]), name = 'biases')

            hidden3 = tf.nn.relu(tf.matmul(hidden2, self.w3) + self.b3)

        with tf.name_scope('logits'):
            self.b4 = tf.Variable(tf.zeros([self.options.word_vocab_size]), name = 'biases')

        return tf.matmul(hidden3, self.glove_weights, transpose_b=True) + self.b4


    def loss(self, logits, glove_labels):
        #TODO: add mimic loss
        entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, 
                labels=glove_labels, name='xentropy')
        return tf.reduce_mean(entropy, name='xentropy_mean') 

    def optimize(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        return optimizer.minimize(loss,global_step=self.global_step)

    def initialize(self):
        self.session.run([self.init])

    def train(self, inputs, labels):
        for step in xrange(self.options.word_vocab_size//self.options.batch_size):
            _, loss = self.session.run([self._train, self._loss], 
                feed_dict = {self.train_inputs: inputs, 
                    self.train_labels: labels[:, 0]})
            if step % 200 == 0:
                print('Step: ', step, ' loss: ', loss)
        print('Saves model to', self.options.log_dir)

def get_misspellings():
    miss = '../test/test_resources/misspellings.txt'
    with open(miss, 'r') as f:
        res = f.readlines()
        res = [word.split() for word in res]
        tmp = []
        for words in res:
            tmp.append(words[0])
            tmp.append(words[1])
    return tmp

def get_hyper_parameters():
    parser = mimic_utility_classes.GloveParser(FLAGS.glove_file)
    parser.parse()
    glove_weights = parser.get_weights_as_ndarray()
    word_vocab = parser.get_vocabulary()
    options = mimic_utility_classes_Options(FLAGS)
    options.glove_word_emb_dim = parser.get_embedding_dimension() 
    options.word_vocab_size = len(word_vocab)

    dataProcessor = mimic_utility_classes.MimicDataProcessor(
            word_vocab, FLAGS.ngram_order, pad_word=True, 
            batch_size=FLAGS.batch_size, load_data=FLAGS.evaluate, 
            dest=options.log_dir + '/vocab.p')

    options.number_of_ngrams_in_word = dataProcessor.max_length 
    options.ngram_vocab_size = len(dataProcessor.get_ngram_vocab()) 
    
    return options, glove_weights, dataProcessor


def run_training():
    options, glove_weights, dataProcessor = get_hyper_parameters()
    with tf.Session() as session:
        model = MimicGlove(session, options, glove_weights)
        model.initialize()
        for epoch in xrange(FLAGS.epochs):
            inputs, labels = dataProcessor.generate_batches()
            model.train(inputs, labels)
            model.saver.save(session, save_path=os.path.join(model.options.log_dir, "model.ckpt"), global_step=epoch)

def restore():
    options, glove_weights ,dataProcessor = get_hyper_parameters()
    session = tf.Session()
    test_word = dataProcessor.get_ngram_indices_from(['oneword', 'another', 'another'])
    model = MimicGlove(session, options, glove_weights)
    ck = tf.train.latest_checkpoint(model.options.log_dir)
    model.saver.restore(session, save_path=ck)
    print(session.run([model._logits], {model.train_inputs: test_word}))
    session.close()

def main(_):
    print('FLAGS: ', FLAGS)
    if FLAGS.evaluate:
        return restore()
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.1, 
            help='Initial Learning Rate')
    parser.add_argument('--glove_file', type=str, 
            default='test_resources/glove/glove.6B.50d.txt',
            help='File containing glove weights')
    parser.add_argument('--eval_data', type=str, default='test_resource/questions-words.txt',
            help='File consisting of analogies of four tokens')
    parser.add_argument('--epochs', type=int, default=20,
            help='Number of steps to run trainer')
    parser.add_argument('--hidden1', type=int, default=640,
            help='Number of units in first layer')
    parser.add_argument('--hidden2', type=int, default=512,
            help='Number of units in second layer')
    parser.add_argument('--batch_size', type=int, default=256,
            help='Batch_size')
    parser.add_argument('--log_dir', type=str, default='ngrams_log_file',
            help='Directory to put the log data')
    parser.add_argument('--ngram_embed_dim', type=int, default=64,
            help='Ngram embedding size')
    parser.add_argument('--ngram_order', type=int, default=3, 
            help='ngram order to split the words')
    parser.add_argument('--loss', type=str, default='mean squared',
            help='Loss to be used options [mean squared, mimic glove]')
    parser.add_argument('--evaluate', type=bool, default=False,
            help='evaluate')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
