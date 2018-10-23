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


### prepare data

eval_set = 'test'#'valid'#'test'

if eval_set == 'valid':
	source_file = '../data/revised_conll13.input'
	target_file = '../data/revised_conll13.output'
else:
	source_file = '../data/test.x_official.txt'
	target_file = '../data/test.y_official.txt'

eval_source_sen = helper.load_data(source_file).lower().split('\n')
eval_target_sen = helper.load_data(target_file).lower().split('\n')


equal_count = 0
for x,y in zip(eval_source_sen,eval_target_sen):
	if x == y:
		equal_count += 1

print(equal_count)
print(equal_count/len(eval_source_sen))


