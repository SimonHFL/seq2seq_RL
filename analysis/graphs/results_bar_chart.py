import numpy as np

import matplotlib.pyplot as plt

model_names = ["Baseline", "1-gr to word", "2-gr to word", "3-gr to word", "1-gr to 1-gr", "2-gr to 1-gr", "3-gr to 1-gr", "1-gr to 1-gr (PT)", "1-gr to 1-gr (PT)" ]
gleu_scores = [53, 40, 50, 53, 40, 50, 53, 40, 50]
m_scores = [10, 15, 12, 10, 15, 12, 10, 15, 12]


plt.rcParams.update({'axes.labelsize': 'large'})
plt.style.use('seaborn')




ind = np.arange(len(model_names))  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, gleu_scores, width)
rects2 = ax.bar(ind + width, m_scores, width)

# add some text for labels, title and axes ticks
ax.set_ylabel('GLEU Score / $M^2$ Score')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(model_names, rotation=45)

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
