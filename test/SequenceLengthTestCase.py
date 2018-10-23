import sys
sys.path.insert(0, '../')
import os
import unittest
from rnn_model import length, embed_input, encoding_layer
import tensorflow as tf

class SequenceLengthTestCase(unittest.TestCase):

	def test_encoder_is_independent_of_padding(self):

		inputs = [ 	[1,1 ] ]
		inputs_padded = [ [1,1,0,0] ]

		input_data = tf.placeholder(tf.int32, shape=(len(inputs),None))

		embedded_input = embed_input(input_data, source_vocab_size=2, enc_embedding_size=10)

		rnn_size = 10
		num_layers = 1
		keep_prob = tf.placeholder(tf.float32, shape=(), name='keep_prob')
		
		sequence_length = length(input_data)
		encoder_outputs, encoded_state = encoding_layer(embedded_input, rnn_size, num_layers, keep_prob, sequence_length)

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			out, state = sess.run([encoder_outputs, encoded_state], {input_data: inputs, keep_prob: 1.})#, sequence_length: [2] })
			out2, state2 = sess.run([encoder_outputs, encoded_state], {input_data: inputs_padded, keep_prob: 1.})

	def test_sequence_len_func(self):

		inputs = [ 	[1,0,0,0],
					[1,1,0,0],
					[1,1,1,0]  ]

		input = tf.placeholder(tf.int32, shape=(len(inputs),None))
		
		res = length(input)

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			out = sess.run(res, {input: inputs})
			
		for x,y in zip(out, [1,2,3]):
			self.assertEqual(x,y)

if __name__ == '__main__':
	unittest.main()
