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


gleu_calc = GLEU()

eval_set = 'test'#'test'#'test'#'valid'#'test'

if eval_set == 'valid':
	source_file = '../data/revised_conll13.input'
	target_file = '../data/revised_conll13.output'
else:
	source_file = '../data/test.x_official.txt'
	target_file = '../data/test.y_official.txt'

eval_source_sen = helper.load_data(source_file).split('\n')
eval_target_sen = helper.load_data(target_file).split('\n')

mle_cands_file= 'candidates_test_09-11_13-27_MLE_rnn_size256_learning_rate0.0003.txt'
reinf_cands_file = 'candidates_test_12-11_15-44_reinforce_500.txt'
ng_cands_file = 'model_output_without_lm.txt'
mle_cands = helper.load_data(mle_cands_file).lower().split('\n')
reinf_cands = helper.load_data(reinf_cands_file).lower().split('\n')
ng_cands = helper.load_data(ng_cands_file).lower().split('\n')

reinf_cands = reinf_cands[:-1] # last line is empty
mle_cands = mle_cands[:-1] # last line is empty


reinf_gleu = gleu_calc.corpus_gleu(reinf_cands, eval_target_sen, eval_source_sen)
mle_gleu = gleu_calc.corpus_gleu(mle_cands, eval_target_sen, eval_source_sen)
ng_gleu = gleu_calc.corpus_gleu(ng_cands, eval_target_sen, eval_source_sen)

print("reinf_gleu ", reinf_gleu)
print("mle_gleu ", mle_gleu)
print("ng_gleu ", ng_gleu)