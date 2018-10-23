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

    epochs, input_batch_size, rnn_size, num_layers, encoding_embedding_size, decoding_embedding_size, learning_rate, keep_probability, num_samples, per_step_reward = map(experiment.hyperparams.get, ('epochs', 'input_batch_size', 'rnn_size', 'num_layers', 'encoding_embedding_size', 'decoding_embedding_size', 'learning_rate', 'keep_probability', 'num_samples', 'per_step_reward'))
    
    ### prepare data ###
    (train_source_int_text, train_target_int_text), (valid_source_int_text, valid_target_int_text), (
            source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = data_preprocessing.get_data()
  
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

    if experiment.train_method == 'MLE':
        graph_batch_size = input_batch_size
    elif experiment.train_method == 'reinforce' or experiment.train_method == 'reinforce_test':
        graph_batch_size = num_samples

    ### prepare model ###
    tf.reset_default_graph()# maybe need?
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        model = rnn_model.RNN(graph_batch_size, max_source_sentence_length, source_vocab_to_int, target_vocab_to_int, encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers, per_step_reward)
    
    eval_batch_size = 128
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        eval_model = rnn_model.RNN(eval_batch_size, max_source_sentence_length, source_vocab_to_int, target_vocab_to_int, encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers, per_step_reward, False)

    ### train model ###
    if experiment.train_method == 'reinforce_test':
        train.reinforce_test(model, experiment.start_checkpoint, source_vocab_to_int, learning_rate, keep_probability, graph_batch_size, target_int_to_vocab,valid_source, valid_target, per_step_reward)
    else:
        train.train(experiment.name, experiment.train_method, model, epochs, input_batch_size, train_source, train_target, valid_source, valid_target, learning_rate, keep_probability, save_path, experiment.start_checkpoint, target_int_to_vocab, source_vocab_to_int, log_path, graph_batch_size, per_step_reward, experiment.max_hours, eval_model, eval_batch_size)

run_time = datetime.datetime.now().strftime("%d-%m_%H-%M")

def run_mle_exps():

    max_hours = None
    dataset = "en-fr"
    tokenization = "word"
    train_method = 'MLE'
    start_checkpoint = None
    
    hyperparams = {
        'epochs' : 10,
        'input_batch_size' : 128,# 128,
        'rnn_size' : [256, 512],
        'num_layers' : 3,
        'encoding_embedding_size' : 128,
        'decoding_embedding_size' : 128,
        'learning_rate' : [0.0001, 0.0003],
        'keep_probability' : 0.8,
        'num_samples': None
    }

    param_permutations, exp_names = helper.permute_params(hyperparams)

    for params, name in zip(param_permutations, exp_names):
        experiment_name = run_time + "_" + train_method+name
        exp = Experiment(params, train_method, start_checkpoint, dataset, tokenization, experiment_name, max_hours)
        run(exp)

def run_reinf_exps():

    max_hours = None
    #start_checkpoint = tf.train.latest_checkpoint("checkpoints/07-11_14:15_MLE_learning_rate0.0003_rnn_size256")
    start_checkpoint = tf.train.latest_checkpoint("checkpoints/09-11_13-27_MLE_rnn_size256_learning_rate0.0003")
    rnn_size = 256
    train_method = 'reinforce' #'reinforce_test'
    dataset = "en-fr"
    tokenization = "word"

    hyperparams = {
        'epochs' : 1,
        'input_batch_size' : 64,
        'rnn_size' : rnn_size,
        'num_layers' : 3,
        'encoding_embedding_size' : 128,
        'decoding_embedding_size' : 128,
        'learning_rate' : 0.0001,#0.000001,
        'keep_probability' : 0.8,
        'num_samples': 64,
        'per_step_reward': False
    }

    param_permutations, exp_names = helper.permute_params(hyperparams)

    for params, name in zip(param_permutations, exp_names):
        experiment_name = run_time + "_" + train_method+name
        reinforce_exp = Experiment(params, train_method, start_checkpoint, dataset, tokenization, experiment_name, max_hours)
        run(reinforce_exp)

#run_mle_exps()
run_reinf_exps()
