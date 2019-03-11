import os
import pickle
import copy
import random
from collections import Counter


import numpy as np

CODES = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3}


def load_data(path):
    """
    Load Dataset from File
    """
    input_file = os.path.join(path)
    with open(input_file, 'r', encoding='utf-8') as f:
        return f.read()


def create_vocabulary(target_text: str, tokenization: str, most_common=20000) -> (dict, dict):

    word2idx = copy.copy(CODES)
    if tokenization == "word":
        words = [item for sentence in target_text for item in sentence]
        counter = Counter(words)
    elif tokenization == "char":
        chars = [char for word in target_text for char in word]
        counter = Counter(chars)

    for w, _ in counter.most_common(most_common - len(word2idx)):
        word2idx[w] = len(word2idx)
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word


def preprocess_and_save_data(train_source_path, train_target_path, valid_source_path,
                             valid_target_path, text_to_ids, tokenization, save=True, vocab_size=20000,
                             out_file='preprocess.p'):#, enc_ngram_order: int=None, dec_ngram_order: int=None):
    """
    Preprocess Text Data.  Save to to file.
    """

    print('Loading data files....')
    # Preprocess
    train_source_lines = load_data(train_source_path).lower().strip().split("\n")    
    train_target_lines = load_data(train_target_path).lower().strip().split("\n")
    valid_source_lines = load_data(valid_source_path).lower().strip().split("\n")
    valid_target_lines = load_data(valid_target_path).lower().strip().split("\n")

    if tokenization == 'word':
        train_source_tokens = [ s.split() for s in train_source_lines]
        train_target_tokens = [ s.split() for s in train_target_lines]
        valid_source_tokens = [ s.split() for s in valid_source_lines]
        valid_target_tokens = [ s.split() for s in valid_target_lines]

    if tokenization == 'char':
        train_source_tokens = [ [list(w) for w in s.split()][0] for s in train_source_lines]
        train_target_tokens = [ [list(w) for w in s.split()][0] for s in train_target_lines]
        valid_source_tokens = [ [list(w) for w in s.split()][0] for s in valid_source_lines]
        valid_target_tokens = [ [list(w) for w in s.split()][0] for s in valid_target_lines]

    source_word2idx, source_idx2word = create_vocabulary(train_source_tokens, tokenization, most_common=vocab_size) 
    target_word2idx, target_idx2word = create_vocabulary(train_target_tokens, tokenization, most_common=vocab_size)

    train_source_ids, train_target_ids = text_to_ids(train_source_tokens, train_target_tokens,
                                                       source_word2idx, target_word2idx)
    valid_source_ids, valid_target_ids = text_to_ids(valid_source_tokens, valid_target_tokens,
                                                       source_word2idx, target_word2idx)

    if save:
    # Save Data
        with open(out_file, 'wb') as of:
            pickle.dump((
                (train_source_ids, train_target_ids),
                (valid_source_ids, valid_target_ids),
                (source_word2idx, target_word2idx), (source_idx2word, target_idx2word)), of)
    else:
        return train_source_ids, train_target_ids, source_word2idx, source_idx2word

def load_preprocess(file = None):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    if file is None:
        file = 'preprocess.p'
        
    with open(file, mode='rb') as in_file:
        return pickle.load(in_file)


def save_params(params):
    """
    Save parameters to file
    """
    with open('params.p', 'wb') as out_file:
        pickle.dump(params, out_file)


def load_params():
    """
    Load parameters from file
    """
    with open('params.p', mode='rb') as in_file:
        return pickle.load(in_file)


def batch_data(source, target, batch_size):
    """
    Batch source and target together
    """
    # reverse sort source data by source length
    source, target = zip(*sorted(zip(source, target), key=lambda x: len(x[0]), reverse=True))

    # cutoff sentences longer than limit
    sen_len_limit = 50#10#50
    cutoff_idx = 0
    for i, s in enumerate(source):
        if len(s) > sen_len_limit:
            cutoff_idx = i+1
        else:
            break
    source = source[cutoff_idx:]
    target = target[cutoff_idx:]

    # randomize batch order (but same order in each epoch)
    start_idxs = [i for i in range(0,len(source), batch_size)]
    random.seed(1)
    random.shuffle(start_idxs)
    
    for start_i in start_idxs:
        source_batch = source[start_i:start_i + batch_size]
        target_batch = target[start_i:start_i + batch_size]

        # don't yield unfilled batches
        if len(source_batch)<batch_size or len(target_batch)<batch_size:
            continue

        yield np.array(pad_sentence_batch(source_batch)), np.array(pad_sentence_batch(target_batch))


def pad_sentence_batch(sentence_batch):
    """
    Pad sentence with <PAD> id
    """
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [CODES['<PAD>']] * (max_sentence - len(sentence))
            for sentence in sentence_batch]


def permute_params(params):

    all_permutations = [params]
    all_names = ['']

    for k,v in params.items():

        if type(v) is list:
            
            permutations = []
            names = []

            for i,x in enumerate(v):
                
                permutation = [x.copy() for x in all_permutations]
                name = list(all_names)
                        
                for i,d in enumerate(permutation):
                    d[k] = x
                    name[i] += '_' + str(k)+"="+str(x)
                
                permutations+=permutation
                names += name
            
            
            all_permutations = permutations
            all_names = names

    return all_permutations, all_names
