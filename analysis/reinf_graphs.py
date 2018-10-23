import csv
import matplotlib.pyplot as plt

log_file = '12-11_15-44_reinforce_capt.csv'


with open(log_file, 'r') as f:
    reader = csv.reader(f)
    lines = list(reader)

lines = lines[1:]# remove header

x = [l[0] for l in lines]
gleu = [float(l[1])*100 for l in lines]
acc = [float(l[2])*100 for l in lines]


plt.rcParams.update({'axes.labelsize': 'large'})
plt.style.use('seaborn')

fig = plt.figure()
ax = fig.add_subplot(111)

#ax.plot(x, gleu, label='GLEU Score')
ax.plot(x, gleu)
#ax.plot(x, acc, label ='Accuracy (%)')

plt.xlabel('Step')
#plt.ylabel('GLEU Score / Accuracy (%)')
plt.ylabel('GLEU Score')

#plt.legend()
plt.show()
plt.close()

"""
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x, acc)
plt.xlabel('Step')
plt.ylabel('Accuracy')

# manipulate
vals = ax.get_yticks()
ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])
"""

plt.show()
plt.close()

