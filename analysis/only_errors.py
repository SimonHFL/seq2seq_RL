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


def log_eval(line, file): 
    with open(file, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(line)

gleu_calc = GLEU()

eval_set = 'valid'#'test'#'test'#'valid'#'test'

if eval_set == 'valid':
	source_file = '../data/revised_conll13.input'
	target_file = '../data/revised_conll13.output'
else:
	source_file = '../data/test.x_official.txt'
	target_file = '../data/test.y_official.txt'

eval_source_sen = helper.load_data(source_file).lower().split('\n')
eval_target_sen = helper.load_data(target_file).lower().split('\n')

cand_file = 'candidates_valid_09-11_13-27_MLE_rnn_size256_learning_rate0.0003.txt'
#cand_file = 'candidates_valid_12-11_15-44_reinforce_500.txt'
eval_cand_sen = helper.load_data(cand_file).lower().split('\n')
eval_cand_sen = eval_cand_sen[:-1] # last line is empty

# filter sentences without errors 
cands = []
targets = []
sources = []

for c, t, s in zip(eval_cand_sen, eval_target_sen, eval_source_sen):
	if s != t:
		cands.append(c)
		targets.append(t)
		sources.append(s)

corpus_gleu = gleu_calc.corpus_gleu(cands, targets, sources)
print("corp gleu ", corpus_gleu)


