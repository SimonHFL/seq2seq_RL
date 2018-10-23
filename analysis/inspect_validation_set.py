import sys
sys.path.insert(0, '../')

import helper
import data_preprocessing
import tensorflow as tf
import numpy as np
import time
import rnn_model
from train import sentence_to_seq, seq_to_sentence
from utilities.gleu import GLEU
from train import eval_validation_set
import random
import csv

def eval_checkpoint(start_checkpoint, source_seqs, target_seqs, source_sens,target_sens, save_name = None):
	candidates = []
	scores = []
	sources = []
	targets = []

	# create model
	tf.reset_default_graph()
	model = rnn_model.RNN(batch_size, max_source_sentence_length, source_vocab_to_int, target_vocab_to_int, encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers)
	saver = tf.train.Saver()

	with tf.Session() as sess:
		
		saver.restore(sess, start_checkpoint)

		acc, reward, reward_std, reward_ci_low, reward_ci_high, preds= eval_validation_set(source_seqs, target_seqs, batch_size, model, sess, int_to_vocab, max_eval_size)
		
		for i, pred in enumerate(preds):
			sources.append(seq_to_sentence(source_seqs[i], int_to_vocab))
			targets.append(seq_to_sentence(target_seqs[i], int_to_vocab))

			hyp_sent = seq_to_sentence(pred, int_to_vocab).capitalize()
			candidates.append(hyp_sent)
			score = gleu_calc.sentence_gleu(hyp_sent, target_sens[i], source_sens[i])
			scores.append(score)

	print("old avg sen gleu ", reward)
	print("new avg sen gleu ", str(sum(scores)/len(scores)))
	print("accuracy ", acc)
	corpus_gleu = gleu_calc.corpus_gleu(candidates, target_sens, source_sens)
	print("corp gleu ", corpus_gleu)

	copy_corpus_gleu = gleu_calc.corpus_gleu(source_sens, target_sens, source_sens)
	print("copy corp gleu", copy_corpus_gleu)
	old_corpus_gleu = gleu_calc.corpus_gleu(candidates, targets, sources)
	print("old corp gleu", old_corpus_gleu)
	
	if save_name is not None: 
		
		with open('candidates_'+ save_name +'.txt', 'w') as file:
			for item in candidates:
				file.write("%s\n" % item)

		with open('scores_'+ save_name +'.txt', 'w') as file:
			for item in scores:
				file.write("%s\n" % item)

	return corpus_gleu, acc

			
	


def gleu_per_sen_lens(checkpoint, eval_source, eval_target, eval_source_sen, eval_target_sen, exp_name):

	log_file = "sen_lens_" + exp_name + '.csv'

	log_eval(['step', 'gleu', 'accuracy', 'num_sentences'], log_file)

	max_source_sentence_length = max([len(sentence) for sentence in eval_source])
	limits = [5, 10, 15, 20, 50, max_source_sentence_length] # cutoff limit for number of words in sentence

	# reverse sort source data by source length
	eval_source, eval_target, eval_source_sen, eval_target_sen = zip(*sorted(zip(eval_source, eval_target, eval_source_sen, eval_target_sen), key=lambda x: len(x[0]), reverse=True))

	for limit in limits:

		sources = []
		references = []
		candidates = []
		scores = []
		
		eval_source_limited = eval_source
		eval_target_limited = eval_target 
		eval_source_sen_limited = eval_source_sen
		eval_target_sen_limited = eval_target_sen

		cutoff_idx = 0
		for i, s in enumerate(eval_source_limited):
			if len(s) > limit:
				cutoff_idx = i+1
			else:
				break
		eval_source_limited = eval_source_limited[cutoff_idx:]
		eval_target_limited = eval_target_limited[cutoff_idx:]
		eval_source_sen_limited = eval_source_sen_limited[cutoff_idx:]
		eval_target_sen_limited = eval_target_sen_limited[cutoff_idx:]

		# shuffle
		rnd = random.Random(1234)
		combined = list(zip(eval_source_limited, eval_target_limited, eval_source_sen_limited, eval_target_sen_limited))
		rnd.shuffle(combined)
		eval_source_limited, eval_target_limited, eval_source_sen_limited, eval_target_sen_limited = zip(*combined)

		max_source_sentence_length = max([len(sentence) for sentence in eval_source_limited])

		gleu, acc = eval_checkpoint(checkpoint, eval_source_limited, eval_target_limited, eval_source_sen_limited, eval_target_sen_limited, exp_name+"_"+str(limit))

		log_eval([limit, gleu, acc, len(eval_source_limited)], log_file)


def log_eval(line, file): 
    with open(file, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(line)

def eval_checkpoints(experiment, steps, source_seqs, target_seqs, source_sens, target_sens):
	print(experiment)
	log_file = experiment + '_capt.csv'

	log_eval(['step', 'gleu', 'accuracy'], log_file)

	for step in steps:
		print(step)
		checkpoint = "../checkpoints/"+experiment+"/dev-"+str(step)
		gleu, acc = eval_checkpoint(checkpoint, source_seqs, target_seqs, source_sens,target_sens)
		
		log_eval([step, gleu, acc], log_file)

### prepare data

eval_set = 'valid'#'test'#'test'#'valid'#'test'

if eval_set == 'valid':
	source_file = '../data/revised_conll13.input'
	target_file = '../data/revised_conll13.output'
else:
	source_file = '../data/test.x_official.txt'
	target_file = '../data/test.y_official.txt'

#eval_source_sen = helper.load_data(source_file).lower().split('\n')
#eval_target_sen = helper.load_data(target_file).lower().split('\n')
eval_source_sen = helper.load_data(source_file).split('\n')
eval_target_sen = helper.load_data(target_file).split('\n')

test_source, test_target = data_preprocessing.get_test(base_path="../")

(_, _), (valid_source, valid_target), (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = data_preprocessing.get_data(pickle_path="../")

if eval_set == 'test':
	eval_source = test_source
	eval_target = test_target
else: 
	eval_source = valid_source
	eval_target = valid_target

int_to_vocab = source_int_to_vocab # target vocab is the same as source vocab

max_eval_size = None

### prepare model 
batch_size = 128
rnn_size = 256
num_layers = 3
encoding_embedding_size = 128
decoding_embedding_size = 128

gleu_calc = GLEU()

max_source_sentence_length = max([len(sentence) for sentence in eval_source])


start_checkpoint = tf.train.latest_checkpoint("../checkpoints/09-11_13-27_MLE_rnn_size256_learning_rate0.0003")
save_name = eval_set + "_" + "09-11_13-27_MLE_rnn_size256_learning_rate0.0003_capt"

"""
exp= "12-11_15-44_reinforce"
step = 500
start_checkpoint = "../checkpoints/"+exp+"/dev-"+str(step)
save_name = eval_set + "_" + exp + "_" + str(step) + "_capt"
"""
eval_checkpoint(start_checkpoint, eval_source, eval_target, eval_source_sen, eval_target_sen, save_name)

#eval_checkpoints('12-11_15-44_reinforce',list(range(1000,2100,100)), eval_source, eval_target, eval_source_sen, eval_target_sen)

# start_checkpoint = tf.train.latest_checkpoint("../checkpoints/07-11_14:15_MLE_learning_rate0.0003_rnn_size256")
#gleu_per_sen_lens(start_checkpoint, eval_source, eval_target, eval_source_sen, eval_target_sen, save_name)


	
