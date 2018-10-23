import random
from collections import Counter
from collections import deque
import mimic_utility_defs
import numpy as np
import copy


CODES = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3}

class NgramTextProcessor(object):

    def __init__(self, data, ngram_order, pad_word=True, pad_to_max=True):
        self.ngram_order = ngram_order
        self.pad_word = pad_word
        self.pad_to_max = pad_to_max
        self.ngram2idx, self.idx2ngram = self._get_ngram_vocabulary(data)

    def _get_ngram_vocabulary(self, data):
        data = ' '.join([sentences for sentences in data.split('\n')])
        words = data.split()
        n2i = copy.copy(CODES)
        counter = Counter()
        for word in words:
            counter.update(self._make_word_ngram(word))
        for ngram, _ in counter.most_common():
            n2i[ngram] = len(n2i)
        i2n = {val: key for key, val in n2i.items()}
        return n2i, i2n

    def _make_word_ngram(self, word):
        word = "%s%s%s"%('$', word, '$') if self.pad_word else word
        #In case the ngram order is larger than the given word length we extra pad the word
        for _ in range(self.ngram_order - len(word)):
            word = "%s%s" % (word, '$')
        return [word[idx:idx+self.ngram_order] for idx in range(0, len(word) + 1 - self.ngram_order)]

    def text_to_ngram_indices(self, text: str):
        sentences = text.split('\n')
        max_len_word = mimic_utility_defs.len_of_longest_word(sentences)
        max_len_word = max_len_word + 2 if self.pad_word else max_len_word
        ngram_indices = [self._sentence_to_ngram_indices(sentence, max_len_word) for sentence in sentences]
        return np.array(ngram_indices)

    def _sentence_to_ngram_indices(self, sentence, max_len_word):
        return [self._word_to_ngram_indices(word, max_len_word) for word in sentence.split(' ') if word != '']

    def _word_to_ngram_indices(self, word, max_len_word):
        res = [self.ngram2idx.get(ngram, self.ngram2idx['<UNK>']) for ngram in self._make_word_ngram(word)]
        if self.pad_to_max:
            pad_id = self.ngram2idx['<PAD>']
            res.extend([pad_id for _ in range(max_len_word - len(res))])
        return res

class DataPreprocessor(object):

    def __init__(self, data, batch_size=16, ngram=1, pad_word=True, pad_to_max=True, lower_word=True, window=2, verbose=False, load_data=False, dest='vocab.p'):
        data = ' '.join([sentences.lower().strip('.') if lower_word else sentences for sentences in data.split('\n')])
        self._words = data.split()
        self.ngram = ngram
        self.pad_word = pad_word
        self.pad_to_max = pad_to_max 
        self.batch_size = batch_size 
        self.window = window
        if load_data:
            self.ngram2idx, self.idx2ngram, self.word2idx, self.idx2word = self._load_vocab(dest)
        else:
            self.ngram2idx, self.idx2ngram, self.word2idx, self.idx2word = self._create_vocab()
            self._save_vocab(dest)
        self.max_length = self._get_max_ngram_length()
        if verbose:
            print('Number of words: {} '.format(len(self._words)))
            print('Sample: {}'.format(self._words[:50]))
            print("Number of unique: {}".format(len(self.word2idx)))
        self._unk_id = self.ngram2idx['<UNK>']
        self._pad_id = self.ngram2idx['<PAD>']
        self.data_index = 0
        self.inputs = [0 for _ in range(self.batch_size)] 
        self.labels = [0 for _ in range(self.batch_size)] 
        self.span = 2 * self.window + 1
        self.num_skips = 2 * self.window
        assert self.batch_size % self.num_skips == 0
        assert self.num_skips <= 2 * self.window
        self.buf = deque(maxlen=self.span)

    
    def _get_max_ngram_length(self):
        return max(len(self._make_word_ngram(word)) for word in self.word2idx.keys())

    def _create_vocab(self):
        w2i = copy.copy(CODES)
        n2i = copy.copy(CODES)
        counter = Counter()
        for word in self._words:
            if word not in w2i:
                w2i[word] = len(w2i)
            counter.update(self._make_word_ngram(word))
        for ngram, _ in counter.most_common():
            n2i[ngram] = len(n2i)
        i2n = self._swap(n2i)
        i2w = self._swap(w2i)
        return n2i, i2n, w2i, i2w

    def _swap(self, d):
        return {val:key for key, val in d.items()}

    def _load_vocab(self, dest):
        return mimic_utility_defs.load(dest)

    def _save_vocab(self, dest):
        mimic_utility_defs.dump((self.ngram2idx, self.idx2ngram, 
                self.word2idx, self.idx2word) , dest)

    def generate_batch(self):
        if self.data_index + self.span > len(self._words):
            self.data_index = 0
        self.buf.extend(self._words[self.data_index: self.data_index+self.span])
        self.data_index += self.span
        for i in range(self.batch_size // self.num_skips):
            target = self.window 
            targets_to_avoid = [self.window]
            for j in range(self.num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, self.span -1)
                targets_to_avoid.append(target)
                self.inputs[i * self.num_skips + j] = self.buf[self.window]
                self.labels[i * self.num_skips + j] = self.buf[target]
            if self.data_index == len(self._words):
                self.buf.extend(self._words[0:self.span])
                self.data_index = self.span
            else:
                self.buf.append(self._words[self.data_index])
                self.data_index += 1
        self.data_index = (self.data_index + len(self._words) - self.span) % len(self._words)
        return _get_np_arrays_for([self._word_to_ngram_index(word) for word in self.inputs], [self.word2idx[label] for label in self.labels])


    def _word_to_ngram_index(self, word):
        word_ngram = self._make_word_ngram(word)
        res = [self.ngram2idx[ng] if ng in self.ngram2idx else self._unk_id for ng in word_ngram]
        if self.pad_to_max:
            res.extend([self._pad_id for _ in range(self.max_length - len(word_ngram))])
        return res

    def _make_word_ngram(self, word):
        word = "%s%s%s"%('$',word,'$') if self.pad_word else word
        for _ in range(self.ngram - len(word)):
            word = "%s%s"%(word, '$')
        return [word[idx:idx+self.ngram] for idx in range(0, len(word) + 1 - self.ngram)]

    def _ngramidx_to_word(self, ngrams):
        return ''.join([self.idx2ngram[ngram] for ngram in ngrams])

    def _wordidx_to_word(self, wordidx):
        return self.idx2word[wordidx]

def _get_np_arrays_for(inputs, labels):
    np_inputs = np.array(np.array([inp for inp in inputs], dtype=np.int32))
    np_labels = np.array(labels, dtype=np.int32).reshape((len(labels), 1))
    return np_inputs, np_labels

class GloveParser(object):
    
    def __init__(self, glove_weights_file):
        self.file = glove_weights_file

    def parse(self) -> "Creates the dictionary and loads the weigths. Return None":
        with open(self.file) as fl:
            words = fl.readlines()
        res = [self._create_word_from_line(word) for word in words]
        word2idx = {}
        word_idx2weights = {}
        for word, weights in res:
            word2idx[word] = len(word2idx)
            word_idx2weights[word2idx[word]] = weights
        self.word2idx = word2idx
        self.word_idx2weights = word_idx2weights
        self.idx2word = {idx:word for word, idx in self.word2idx.items()}

    def get_embedding_dimension(self):
        return len(self.word_idx2weights[0])

    def get_vocabulary(self)-> "word to its index dictionary":
        return self.word2idx 

    def get_weights(self):
        return self.word_idx2weights 

    def get_weights_as_ndarray(self):
        tmp = [self.word_idx2weights[idx] for idx in range(len(self.word_idx2weights))]
        return np.array(tmp) 

    def _create_word_from_line(self, line:str) -> ("word", "list with word weights"):
        res = line.split()
        word, weights = res[0], res[1:]
        weigths = map(float, weights)
        return (word, weights)

class MimicDataProcessor(object):

    def __init__(self, vocabulary:dict, ngram_order:int=1, pad_word:bool=True, pad_to_max:bool=True, batch_size:int=16, load_data=False, dest="default.p"):
        self.word_vocabulary= vocabulary
        self.dataProcessor = DataPreprocessor('\n'.join(self.word_vocabulary.keys()),
                ngram=ngram_order, pad_word=pad_word, pad_to_max=pad_to_max, load_data=load_data, dest=dest)
        self.batch_size = batch_size
        self.max_length = self.dataProcessor.max_length
        self.batcher = LinkedBatcher(list(self.word_vocabulary.keys()), batch_size=batch_size)

    def generate_batches(self):
        words = self.batcher.get_batch()
        inputs = [self.dataProcessor._word_to_ngram_index(word) for word in words]
        labels = [self.word_vocabulary[word] for word in words]
        return np.array(inputs, dtype=np.int32), np.array(labels, dtype=np.int32).reshape((len(labels), 1))

    def get_ngram_vocab(self):
        return self.dataProcessor.ngram2idx

    def get_index_to_ngram(self):
        return self.dataProcessor.idx2ngram

    def read_analogies(self, analogy_file):
        """Similar to read_analogies() in:
        https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec.py
        """
        a = []
        b = []
        c = []
        d = []
        questions_skipped = 0 
        with open(analogy_file, "r") as analogy_f:
            for line in analogy_f:
                if line.startswith(":"): #skip comments
                    continue
                words = line.strip().lower().split(" ")
                words = [word for word in words]
                if len(words) != 4 or words[3] not in self.word_vocabulary:
                    questions_skipped += 1
                else: 
                    a.append(self.dataProcessor._word_to_ngram_index(words[0]))
                    b.append(self.dataProcessor._word_to_ngram_index(words[1]))
                    c.append(self.dataProcessor._word_to_ngram_index(words[2]))
                    d.append(self.word_vocabulary[words[3]])

        print("Eval analogy file: ", analogy_file)
        print("Questions: ", len(a))
        print("Skipped: ", questions_skipped)
        self.n_analogies = len(a)
        self.analogy_questions_with_answers = (np.array(a, dtype=np.int32), np.array(b, dtype=np.int32), np.array(c, dtype=np.int32), np.array(d, dtype=np.int32))

    def get_ngram_indices_from(self, words):
        res = [self.dataProcessor._word_to_ngram_index(word) for word in words]
        return np.array(res)

class LinkedBatcher(object):

    def __init__(self, data:list, batch_size:int):
        if len(data) < batch_size:
            raise ValueError("The batch size should not be greater than then data size")
        self.batch_size = batch_size
        self.pointer = self._create_linked_batch_and_get_head(data)

    def get_batch(self):
        res = []
        pointer = self.pointer
        for _ in range(self.batch_size):
            res.append(pointer.value)
            pointer = pointer.next
        self.pointer = pointer 
        return res 
    
    def _create_linked_batch_and_get_head(self, data):
        head = _Node(data[0], None)
        tmp = head
        tail = data[1:]
        tail.reverse()
        for elem in tail:
            td = _Node(elem, tmp)
            tmp = td
        head.next = tmp
        return head 


class _Node(object):
    
    def __init__(self, value, node):
        self.value = value
        self.next = node

class Options(object):
    """Abstraction of model config"""
    def __init__(self, FLAGS):
        self.ngram_embed_dim = FLAGS.ngram_embed_dim
        self.learning_rate = FLAGS.learning_rate
        self.epochs = FLAGS.epochs
        self.batch_size = FLAGS.batch_size
        self.log_dir = FLAGS.log_dir
        self.hidden1 = FLAGS.hidden1
        self.hidden2 = FLAGS.hidden2
        self.evaluate = FLAGS.evaluate
        self.loss = FLAGS.loss
        self.ngram_order = FLAGS.ngram_order
