import sys
sys.path.insert(0, '../')

from data_preprocessing import get_vocab
import numpy as np
import matplotlib.pyplot as plt
import csv


def log_eval(line, file): 
    with open(file, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(line)


max_vocab_size = 200000

log_file = 'vocab_coverage_full.csv'

def calc_coverage():
	for vocab_size in range(1000,max_vocab_size, 1000):
		print("vocab size ", vocab_size)
		source_text, target_text, word2idx, idx2word = get_vocab('../data/train.x_official.txt','../data/train.y_official.txt','../data/revised_conll13.input', '../data/revised_conll13.output', vocab_size)
		unk_idx = word2idx['<UNK>']
		print("loaded")
		num_source_unks = 0
		num_source_words = 0
		num_target_unks = 0
		num_target_words = 0
		for source_sen, target_sen in zip(source_text, target_text):
			num_source_words += len(source_sen)
			for word_idx in source_sen:
				if word_idx == unk_idx:
					num_source_unks +=1
			num_target_words += len(target_sen)
			for word_idx in target_sen:
				if word_idx == unk_idx:
					num_target_unks +=1
		source_unk_perc = num_source_unks/num_source_words
		target_unk_perc = num_target_unks/num_target_words
		print("source unk percentage:", source_unk_perc)
		print("target unk percentage:", target_unk_perc)
		log_eval([vocab_size, source_unk_perc, target_unk_perc], log_file)

#calc_coverage()
# load precalc

with open(log_file, 'r') as f:
    reader = csv.reader(f)
    coverages = list(reader)

x = [i[0] for i in coverages]
y_source = [i[1] for i in coverages]
y_target = [i[2] for i in coverages]


plt.rcParams.update({'axes.labelsize': 'large'})
plt.style.use('seaborn')


fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x, y_source, label="source")
ax.plot(x, y_target, label="target")
plt.xlabel('Vocabulary Size')
plt.ylabel('Percentage Unknown Words')

# manipulate
vals = ax.get_yticks()
ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])

plt.legend()
plt.show()

print(y_source)


"""






plt.axis([0, max_vocab_size, 0, 0.2])
plt.ion()

plt.scatter(vocab_size, source_unk_perc)
plt.scatter(vocab_size, target_unk_perc)
plt.pause(0.05)
"""
