
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
import matplotlib.pyplot as plt

# load source target
# load candidates
# score candidates
# diffs

(_, _), (valid_source, valid_target), (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = data_preprocessing.get_data(pickle_path="../")

gleu_calc = GLEU()


source_file = '../data/test.x_official.txt'
target_file = '../data/test.y_official.txt'

eval_source_sen = helper.load_data(source_file).split('\n')
eval_target_sen = helper.load_data(target_file).split('\n')

mle_cands_file= 'candidates_test_09-11_13-27_MLE_rnn_size256_learning_rate0.0003_capt.txt'
#other_cands_file = 'baseline_source_ngram1_to_word_test_candidates.txt'
other_cands_file = 'final_ngram1_ngram1_test_candidates.txt'
mle_cands = helper.load_data(mle_cands_file).split('\n')
other_cands = helper.load_data(other_cands_file).split('\n')

#reinf_cands = reinf_cands[:-1] # last line is empty
mle_cands = mle_cands[:-1] # last line is empty



diffs = []
# compute scores
for mle_cand, other_cand, t, s in zip(mle_cands, other_cands, eval_target_sen, eval_source_sen):

	mle_score = gleu_calc.sentence_gleu(mle_cand, t, s)
	other_score = gleu_calc.sentence_gleu(other_cand, t, s)
	diff = other_score - mle_score
	diffs.append(diff)


for diff, other_cand, mle_cand, t, s in zip(diffs, other_cands, mle_cands, eval_target_sen, eval_source_sen):
	
	#if s != "When we are diagonosed out with certain genetic disease , are we suppose to disclose this result to our relatives ?":
	#	continue
	if s == t: # not so interesting
		continue

	s_seq = sentence_to_seq(s.lower(), source_vocab_to_int)
	t_seq = sentence_to_seq(t.lower(), source_vocab_to_int)

	baseline_words = mle_cand.split()
	other_words = other_cand.split()
	source_words = s.split()
	target_words = t.split()

	# input unk
	# target unk
	# ng correct target

	spelling_mistake = False
	for i, (s_t, w_t) in enumerate(zip(s_seq, t_seq)):
		if i > len(s_seq)-1 or i > len(t_seq)-1 or i > len(target_words)-1 or i > len(other_words)-1:
			continue
		if s_t == 2 and w_t == 2:
			#print(i)
			#print(len(target_words), len(other_words))
			
			if target_words[i] in other_words and target_words[i-1] in other_words:# and target_words[i+1] in other_words:# other_words[i] or target_words[i] == other_words[i+1] or target_words[i] == other_words[i-1] :
				spelling_mistake = True
				print("-"*10)
				print(s)
				print(seq_to_sentence(s_seq, source_int_to_vocab))
				print(t)
				print(seq_to_sentence(t_seq, source_int_to_vocab))
				print(mle_cand)
				print(other_cand)

			"""
			if t_x in other_cand.split():
				print("-"*10)
				print("source ", s)
				print("source dec", seq_to_sentence(s_seq, source_int_to_vocab))
				#print("target ", t_x)
				print("candidate", other_cand)
			"""
			
			
	continue
	if not spelling_mistake:
		continue


	#if diff>0.2 and diff < 0.3:
	print("-"*10)
	print(diff)
	print("source ", s)
	print("source dec", seq_to_sentence(s_seq, source_int_to_vocab))
	print("target ", t)
	print("other_cand ", other_cand)
	print("mle_cand ", mle_cand)



		

