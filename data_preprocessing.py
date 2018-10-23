import helper
import numpy as np
import os

def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
    Convert source and target text to proper word ids
    :param source_text: String that contains all the source text.
    :param target_text: String that contains all the target text.
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: A tuple of lists (source_id_text, target_id_text)
    """
    # TODO: Implement Function

    source_sentences = source_text.split("\n")
    source_sentences = [sentence.split(" ") for sentence in source_sentences]
    # remove '' to avoid errors
    source_sentences = [[word for word in sentence if word != ''] for sentence in source_sentences]
    source_id_text = [[source_vocab_to_int[word if word in source_vocab_to_int else '<UNK>'] for word in sentence] for sentence in source_sentences]
    
    target_sentences = target_text.split("\n")
    target_sentences = [sentence.split(" ") for sentence in target_sentences]
    # remove '' to avoid errors
    target_sentences = [[word for word in sentence if word != ''] for sentence in target_sentences]
    target_id_text = [ [ target_vocab_to_int[word if word in target_vocab_to_int else '<UNK>'] for word in sentence] for sentence in target_sentences]
    target_id_text = [ sentence + [target_vocab_to_int['<EOS>']] for sentence in target_id_text]    
    #target_id_text = [ [target_vocab_to_int['<GO>']] + sentence + [target_vocab_to_int['<EOS>']] for sentence in target_id_text]    
    #target_id_text = [ [target_vocab_to_int['<GO>']] + sentence for sentence in target_id_text]    

    return (source_id_text, target_id_text)

def get_data(folder, train_source_file, train_target_file, dev_source_file, dev_target_file, pickle_path = ""):


    #pickle_dump =  "preprocess_{}.p".format(
    #    "word" if word_token else "ngram_" + str(enc_ngram_order_tokenization))
    pickle_dump = os.path.join(folder, "preprocessed.pickle")
    
    if not os.path.exists(pickle_dump):
        helper.preprocess_and_save_data(
            os.path.join(folder, train_source_file),
            os.path.join(folder, train_target_file), 
            os.path.join(folder, dev_source_file),
            os.path.join(folder, dev_target_file), 
            text_to_ids,
            out_file=pickle_dump)
    else:
        print('Loads datasets from {}'.format(pickle_dump))
    return helper.load_preprocess(file=pickle_dump)


def get_test(test_source_file='data/test.x_official.txt',
             test_target_file='data/test.y_official.txt',
             base_path="",
             pickle=None):
    pickle = base_path + 'preprocess_word.p' if pickle is None else pickle

    _, _, (_, target_vocab_to_int), \
    (_, target_int_to_vocab) = helper.load_preprocess(pickle)

    test_source_text = helper.load_data(base_path+test_source_file).lower()
    test_target_text = helper.load_data(base_path+test_target_file).lower()
    return text_to_ids(test_source_text, test_target_text, target_vocab_to_int, target_vocab_to_int)


def get_vocab(source_path, target_path, valid_source_path, valid_target_path, vocab_size):

    source_text, target_text, word2idx, idx2word = helper.preprocess_and_save_data(source_path, target_path, valid_source_path, valid_target_path, text_to_ids, save=False, vocab_size=vocab_size)
    
    return source_text, target_text, word2idx, idx2word
