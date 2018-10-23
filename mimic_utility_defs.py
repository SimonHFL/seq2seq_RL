import pickle
import numpy as np

from sklearn.manifold import TSNE
import os

def dump(obj, f):
    with open(f, 'wb') as out:
        pickle.dump(obj, out)

def load(f):
    with open(f, 'rb') as in_f:
        return pickle.load(in_f)

def reduce_dimensions_and_draw(l2v: dict, filename='tsne.png'):
    l2v = sorted(l2v.items())
    vectors = [v for _, v in l2v]
    labels = [l for l, _ in l2v]
    print(labels)
    reduced_v = _reduce_v(vectors)
    _plot_with_labels(labels, reduced_v, filename)

def _reduce_v(vectors):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    return tsne.fit_transform(vectors)

def _plot_with_labels(labels, vectors, filename):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    '''Credits for the 4 following lines goes to: https://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined 
    author: Matthias123'''
    if os.environ.get('DISPLAY', '') == '':
        print('no display found. Using Agg backend')
        mpl.use('Agg')
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = vectors[i, :]
        plt.scatter(x,y)
        plt.annotate(label,
                xy=(x,y),
                xytext=(5,2),
                textcoords='offset points',
                ha='right',
                va='bottom')
        plt.savefig(filename)

def len_of_longest_word(sentences: list):
    mx = -1
    for sentence in sentences:
        for word in sentence.split(' '):
            l = len(word)
            mx = l if l > mx else mx
    return mx
