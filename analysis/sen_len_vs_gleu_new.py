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

mle_cands_file= 'candidates_test_09-11_13-27_MLE_rnn_size256_learning_rate0.0003.txt'
reinf_cands_file = 'candidates_test_12-11_15-44_reinforce_500.txt'
mle_cands = helper.load_data(mle_cands_file).split('\n')
reinf_cands = helper.load_data(reinf_cands_file).split('\n')

reinf_cands = reinf_cands[:-1] # last line is empty
mle_cands = mle_cands[:-1] # last line is empty

# capitalize first
reinf_cands = [x.capitalize() for x in reinf_cands]
mle_cands = [x.capitalize() for x in mle_cands]



full_mle_gleu = gleu_calc.corpus_gleu(mle_cands, eval_target_sen, eval_source_sen)
full_reinf_gleu = gleu_calc.corpus_gleu(reinf_cands, eval_target_sen, eval_source_sen)
print("full_mle_gleu ", full_mle_gleu)
print("full_reinf_gleu ", full_reinf_gleu)

bucket_sizes = []

# reverse sort source data by source length
eval_source_sen, eval_target_sen, reinf_cands, mle_cands = zip(*sorted(zip(eval_source_sen, eval_target_sen, reinf_cands, mle_cands), key=lambda x: len(x[0].split()), reverse=False))
all_sources = []
sources = []
targets = []
reinf_candidates = []
mle_candidates = []

mle_scores = []
reinf_scores = []

max_source_sentence_length = max([len(sentence.split()) for sentence in eval_source_sen])


sen_lens = np.array([len(x.split()) for x in eval_source_sen])
quartiles = np.percentile(sen_lens, np.arange(0, 100, 25)).tolist()
deciles = np.percentile(sen_lens, np.arange(0, 100, 10)).tolist()
print(quartiles)

limits = quartiles[1:] + [max_source_sentence_length]#[10, 15, 20, 30, 40, 50, max_source_sentence_length]
l = 0 
for s, t, r, m in zip(eval_source_sen, eval_target_sen, reinf_cands, mle_cands):
	if len(s.split()) <= limits[l]:
		sources.append(s)
		targets.append(t)
		reinf_candidates.append(r)
		mle_candidates.append(m)
	else:
		sources.append(s)
		targets.append(t)
		reinf_candidates.append(r)
		mle_candidates.append(m)
		all_sources += sources
		#print("limit ", limits[l])
		#print('num sents ',len(sources))
		mle_gleu = gleu_calc.corpus_gleu(mle_candidates, targets, sources)
		reinf_gleu = gleu_calc.corpus_gleu(reinf_candidates, targets, sources)
		#print(mle_gleu)
		#print(reinf_gleu)
		mle_scores.append(mle_gleu)
		reinf_scores.append(reinf_gleu)
		bucket_sizes.append(len(sources))
		sources = []
		targets = []
		reinf_candidates = []
		mle_candidates = []

		l += 1


		# evaluate
		# clear
		# increase limit
# evaluate last

print("limit ", limits[-1])
print('num sents ',len(sources))
bucket_sizes.append(len(sources))
mle_gleu = gleu_calc.corpus_gleu(mle_candidates, targets, sources)
reinf_gleu = gleu_calc.corpus_gleu(reinf_candidates, targets, sources)
print(mle_gleu)
print(reinf_gleu)
mle_scores.append(mle_gleu)
reinf_scores.append(reinf_gleu)
all_sources += sources


#fix gleu format
mle_scores = [x*100 for x in mle_scores]
reinf_scores = [x*100 for x in reinf_scores]

print(bucket_sizes)
print(sum(bucket_sizes))
print("out of ", len(eval_source_sen))

weights = [ x / sum(bucket_sizes) for x in bucket_sizes]
weighted_mle_gleu = sum([x*y for x,y in zip(mle_scores, weights)])
weighted_reinf_gleu = sum([x*y for x,y in zip(reinf_scores, weights)])
print("weighted mle gleu ", weighted_mle_gleu)
print("weighted reinf gleu ", weighted_reinf_gleu)

#print(all_sources)
diff = list(set(eval_source_sen) - set(all_sources))
print([len(x.split()) for x in diff])
print(diff)



plt.rcParams.update({'axes.labelsize': 'large'})
plt.style.use('seaborn')
"""
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(limits, mle_scores, label='Baseline (MLE)')
ax.plot(limits, reinf_scores, label='REINFORCE')

plt.xlabel('Max Words per Sentence')
plt.ylabel('GLEU Score')

plt.legend()

plt.show()
plt.close()
"""


print(limits)
print("reinf_scores", reinf_scores)
print("mle_scores", mle_scores)

ind = np.arange(len(limits))  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, mle_scores, width)


rects2 = ax.bar(ind + width, reinf_scores, width)

# add some text for labels, title and axes ticks
ax.set_ylabel('GLEU Score')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('1-16 Words', '17-21 Words', '22-28 Words', '29-227 Words'))

ax.legend((rects1[0], rects2[0]), ('MLE', 'RL'))

"""
def autolabel(rects):

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
"""

plt.show()
