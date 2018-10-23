import sys
sys.path.insert(0, '../')
import helper
import numpy as np
from matplotlib import pyplot as plt
import numpy as np

#plt.style.use('ggplot')
(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = helper.load_preprocess("../preprocess_20k.p")

source_lens = np.array([len(x) for x in source_int_text])

print(source_lens.mean())
print(source_lens.std())

target_lens = np.array([len(x) for x in target_int_text])
print(target_lens.mean())
print(target_lens.std())

print(str(source_lens.mean()+target_lens.mean()/2))


"""
source_lens.sort()
print(source_lens[:100])
for i, s in enumerate(source_lens):
	if s>50:
		break
print(i)
print((i+1)/len(source_lens))
"""

plt.style.use('seaborn')
plt.rcParams.update({'axes.labelsize': 'large'})

fig_per_hour = plt.figure()
per_hour = fig_per_hour.add_subplot(111)
counts, bins, patches = per_hour.hist(
    source_lens, bins = 100, normed = False,linewidth=0)

#counts, bins, patches = per_hour.hist(
#    target_lens, bins = 100, normed = False,linewidth=0)


plt.gca().set_xlim(source_lens.min(), source_lens.max())

per_hour.set_xlabel('Words per Sentence')
per_hour.set_ylabel('Number of Sentences')
#fig_per_hour.suptitle('test title', fontsize=20)
fig_per_hour.savefig('sen_len_histogram.jpg')


plt.show()
