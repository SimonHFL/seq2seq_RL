import csv
import matplotlib.pyplot as plt

log_file = 'sen_lens_test_09-11_13-27_MLE_rnn_size256_learning_rate0.0003.csv'
log_file_2 = 'sen_lens_test_12-11_15-44_reinforce_500.csv'

with open(log_file, 'r') as f:
    reader = csv.reader(f)
    lines = list(reader)

with open(log_file_2, 'r') as f:
    reader = csv.reader(f)
    lines_2 = list(reader)

#lines = lines[1:]# remove header
lines = lines[1:]# remove header
lines_2 = lines_2[1:]# remove header

x = [l[0] for l in lines]
gleu_mle = [float(l[1])*100 for l in lines]
gleu_reinf = [float(l[1])*100 for l in lines_2]
print(gleu_reinf)
diff = len(x) - len(gleu_reinf)
print(diff)
if diff > 0:
	pad = [0] * diff
	gleu_reinf += pad
print(gleu_reinf)

plt.rcParams.update({'axes.labelsize': 'large'})
plt.style.use('seaborn')

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x, gleu_mle, label='Baseline (MLE)')
ax.plot(x, gleu_reinf, label='REINFORCE')

plt.xlabel('Max Words per Sentence')
plt.ylabel('GLEU Score')

plt.legend()

plt.show()
plt.close()

