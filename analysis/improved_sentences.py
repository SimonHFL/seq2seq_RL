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


def log_eval(line, file): 
    with open(file, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(line)

gleu_calc = GLEU()

eval_set = 'test'#'test'#'test'#'valid'#'test'

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

mle_cands_file= 'candidates_test_09-11_13-27_MLE_rnn_size256_learning_rate0.0003_capt.txt'
reinf_cands_file = 'candidates_test_12-11_15-44_reinforce_500_capt.txt'
mle_cands = helper.load_data(mle_cands_file).split('\n')
reinf_cands = helper.load_data(reinf_cands_file).split('\n')

reinf_cands = reinf_cands[:-1] # last line is empty
mle_cands = mle_cands[:-1] # last line is empty

# capitalize first letter
#reinf_cands = [x.capitalize() for x in reinf_cands]
#mle_cands = [x.capitalize() for x in mle_cands]


# load scores

mle_scores_file= 'scores_test_09-11_13-27_MLE_rnn_size256_learning_rate0.0003_capt.txt'
reinf_scores_file = 'scores_test_12-11_15-44_reinforce_500_capt.txt'
mle_scores = helper.load_data(mle_scores_file).lower().split('\n')
reinf_scores = helper.load_data(reinf_scores_file).lower().split('\n')

reinf_scores = reinf_scores[:-1] # last line is empty
mle_scores = mle_scores[:-1] # last line is empty

# load scores

diffs = []

for reinf_score, mle_score, reinf_cand, mle_cand, t, s in zip(reinf_scores, mle_scores, reinf_cands, mle_cands, eval_target_sen, eval_source_sen):
	
	if s == t: # not so interesting
		continue
	diff = float(reinf_score)-float(mle_score)
	diffs.append(diff)
	"""
	if diff>0.2 and diff < 0.3:
		print("-"*10)
		print(diff)
		print("source ", s)
		print("target ", t)
		print("reinf_cand ", reinf_cand)
		print("mle_cand ", mle_cand)
	"""
	

print(sum([1 for x in diffs if x == 0  ]))
#quartiles = np.percentile(diffs, np.arange(0, 100, 25)).tolist()
#deciles = np.percentile(diffs, np.arange(0, 110, 10)).tolist()
#print(deciles)
#exit()

# correct diffs scale
diffs = [x*100 for x in diffs]

plt.style.use('seaborn')
plt.rcParams.update({'axes.labelsize': 'large'})

fig_per_hour = plt.figure()
per_hour = fig_per_hour.add_subplot(111)
counts, bins, patches = per_hour.hist(
    diffs, bins = 100, normed = False,linewidth=0, align='left')

#counts, bins, patches = per_hour.hist(
#    target_lens, bins = 100, normed = False,linewidth=0)


plt.gca().set_xlim(-100, 100)

#plt.xlabel('GLEU Score Difference')
plt.xlabel('$I_{RL}$')
plt.ylabel('Number of Sentences')


#fig_per_hour.suptitle('test title', fontsize=20)



plt.show()		

