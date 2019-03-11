import tensorflow as tf
import helper
import data_preprocessing
import rnn_model
import train
from distutils.version import LooseVersion
import warnings
from experiment import Experiment
import os
import shutil
import git
import random
from functools import reduce
import datetime
#import Levenshtein
import textdistance

def run(experiment):
    save_path = "checkpoints/" + experiment.name 
    log_path = "tensorboard/train/" + experiment.name
    # create or clean directory
    for path in [save_path, log_path]:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            shutil.rmtree(path)           
            os.makedirs(path)
    save_path += "/dev"

    # log git commit hash
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    file = open(log_path + "/git_commit_" + sha, 'w')
    file.close()

    epochs, input_batch_size, rnn_size, num_layers, encoding_embedding_size, decoding_embedding_size, learning_rate, keep_probability, num_samples, reward = map(experiment.hyperparams.get, ('epochs', 'input_batch_size', 'rnn_size', 'num_layers', 'encoding_embedding_size', 'decoding_embedding_size', 'learning_rate', 'keep_probability', 'num_samples', "reward"))
    
    ### prepare data ###
    (train_source_int_text, train_target_int_text), (valid_source_int_text, valid_target_int_text), (
            source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = data_preprocessing.get_data(experiment.data["dataset"], experiment.data["folder"], experiment.data["train_source_file"], experiment.data["train_target_file"], experiment.data["dev_source_file"], experiment.data["dev_target_file"], experiment.tokenization)

    max_source_sentence_length = max([len(sentence) for sentence in train_source_int_text])

    train_source = train_source_int_text
    train_target = train_target_int_text
    
    valid_source = valid_source_int_text
    valid_target = valid_target_int_text

    # shuffle
    rnd = random.Random(1234)
    train_combined = list(zip(train_source, train_target))
    rnd.shuffle(train_combined)
    train_source, train_target = zip(*train_combined)

    valid_combined = list(zip(valid_source, valid_target))
    rnd.shuffle(valid_combined)
    valid_source, valid_target = zip(*valid_combined)

    # set reward function
    if reward == "levenshtein":
        reward_func = lambda ref_hyp: - textdistance.levenshtein(ref_hyp[0], ref_hyp[1])   
    elif reward == "jaro-winkler":
        reward_func = lambda ref_hyp: textdistance.JaroWinkler()(ref_hyp[0], ref_hyp[1]) 
    elif reward == "hamming":
        reward_func = lambda ref_hyp: - textdistance.hamming(ref_hyp[0], ref_hyp[1])

    if experiment.train_method == 'MLE':
        graph_batch_size = input_batch_size
    elif experiment.train_method == 'reinforce' or experiment.train_method == 'reinforce_test':
        graph_batch_size = num_samples

    ### prepare model ###
    tf.reset_default_graph()# maybe need?
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        model = rnn_model.RNN(graph_batch_size, max_source_sentence_length, source_vocab_to_int, target_vocab_to_int, encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers)
    
    eval_batch_size = 128
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        eval_model = rnn_model.RNN(eval_batch_size, max_source_sentence_length, source_vocab_to_int, target_vocab_to_int, encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers, False)


    early_stopping = True

    ### train model ###
    if experiment.train_method == 'reinforce_test':
        train.reinforce_test(model, experiment.start_checkpoint, source_vocab_to_int, learning_rate, keep_probability, graph_batch_size, target_int_to_vocab, source_int_to_vocab, valid_source, valid_target)
    else:
        train.train(experiment.name, experiment.train_method, model, epochs, input_batch_size, train_source, train_target, valid_source, valid_target, learning_rate, keep_probability, save_path, experiment.start_checkpoint, target_int_to_vocab, source_int_to_vocab, source_vocab_to_int, log_path, graph_batch_size, experiment.max_hours, eval_model, eval_batch_size, reward_func, early_stopping)

run_time = datetime.datetime.now().strftime("%d-%m_%H-%M")
#run_time = "26-02_15-13"
def run_mle_exps(datasets):

    max_hours = None
    tokenization = "char"#"word"
    train_method = 'MLE'
    start_checkpoint = None


    # hyperparams = {
    #     'epochs' : 10,
    #     'input_batch_size' : 16,#128,# 128,
    #     'rnn_size' : 300,
    #     'num_layers' : 1,
    #     'encoding_embedding_size' : 60,
    #     'decoding_embedding_size' : 60,
    #     'learning_rate' : 0.001,#[0.0001, 0.0003],
    #     'keep_probability' : 0.8,
    #     'num_samples': None,
    #     'dataset' : datasets
    # }

    hyperparams = {
        'epochs' : 10,
        'input_batch_size' : 16,#128,# 128,
        'rnn_size' : 256,#[256, 512],
        'num_layers' : 3,
        'encoding_embedding_size' : 128,
        'decoding_embedding_size' : 128,
        'learning_rate' : 0.001,#[0.0001, 0.0003],
        'keep_probability' : 0.8,
        'num_samples': None,
        'dataset' : datasets,
        'reward': "levenshtein"
    }


    # done = ["26-02_15-13_MLE_dataset=english-icamet",
    #         "26-02_15-13_MLE_dataset=german-anselm",
    #         "26-02_15-13_MLE_dataset=hungarian-hgds",
    #         "26-02_15-13_MLE_dataset=icelandic-icepahc"]
    param_permutations, exp_names = helper.permute_params(hyperparams)

    for params, name in zip(param_permutations, exp_names):
        experiment_name = run_time + "_" + train_method+name
        # if experiment_name in done:
        #     continue

        data = {
            "dataset": params['dataset'],
            "folder": "../../Marcel/datasets/preprocessed/",
            "train_source_file": params["dataset"]+'.train.orig',
            "train_target_file": params["dataset"]+'.train.norm',
            "dev_source_file": params["dataset"]+'.dev.orig',
            "dev_target_file": params["dataset"]+'.dev.norm',
        }

        exp = Experiment(params, train_method, start_checkpoint, data, tokenization, experiment_name, max_hours)
        run(exp)

def run_reinf_exps(datasets):

    max_hours = None
    #start_checkpoint = tf.train.latest_checkpoint("checkpoints/07-11_14:15_MLE_learning_rate0.0003_rnn_size256")
    
    
    train_method = 'reinforce' #'reinforce_test'
    tokenization = "char"#"word"





    # data = {
    #     "folder": "data/en-fr/",
    #     "train_source_file": 'train.fr.txt',
    #     "train_target_file": 'train.en.txt',
    #     "dev_source_file": 'dev.fr.txt',
    #     "dev_target_file": 'dev.en.txt',
    # }

    hyperparams = {
        'epochs' : 2,
        'input_batch_size' : 64,
        'rnn_size' : 256,
        'num_layers' : 3,
        'encoding_embedding_size' : 128,
        'decoding_embedding_size' : 128,
        'learning_rate' : 0.00001,#0.0001,#0.000001,
        'keep_probability' : 0.8,
        'num_samples': 64,
        "dataset": datasets,
        'reward': ["levenshtein","jaro-winkler","hamming"]
    }

    # done = ["26-02_15-13_reinforce_reward=hamming_dataset=hungarian-hgds",
    #         "26-02_15-13_reinforce_reward=hamming_dataset=icelandic-icepahc",
    #         "26-02_15-13_reinforce_reward=levenshtein_dataset=hungarian-hgds",
    #         "26-02_15-13_reinforce_reward=levenshtein_dataset=icelandic-icepahc"]

    param_permutations, exp_names = helper.permute_params(hyperparams)

    for params, name in zip(param_permutations, exp_names):

        data = {
            "dataset": params['dataset'],
            "folder": "../../Marcel/datasets/preprocessed/",
            "train_source_file": params['dataset']+'.train.orig',
            "train_target_file": params['dataset']+'.train.norm',
            "dev_source_file": params['dataset']+'.dev.orig',
            "dev_target_file": params['dataset']+'.dev.norm',
        }


        start_checkpoint = tf.train.latest_checkpoint("checkpoints/" + run_time + "_MLE_dataset="+params["dataset"])
        
        experiment_name = run_time + "_" + train_method+name
        # if experiment_name in done:
        #     continue

        reinforce_exp = Experiment(params, train_method, start_checkpoint, data, tokenization, experiment_name, max_hours)
        run(reinforce_exp)



datasets = [
    "hungarian-hgds",
    "icelandic-icepahc",
    "german-anselm",
    "english-icamet",
    #"english-icamet_small",
    #"german-ridges",
    #"portuguese-ps",
    "slovene-goo300k-bohoric",
    #"slovene-goo300k-gaj",
    #"spanish-ps",
    "swedish-gaw",
    ]

run_mle_exps(datasets)
run_reinf_exps(datasets)






