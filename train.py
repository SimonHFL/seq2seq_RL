import helper
import data_preprocessing
import tensorflow as tf
import numpy as np
import time
import rnn_model
import nltk
from nltk.translate.bleu_score import SmoothingFunction
from utilities.gleu import GLEU
import math
import helper
import scipy
import time 
import csv
import textdistance
#from pylev3 import Levenshtein
#import Levenshtein

gleu_calc = GLEU()

def seq_to_sentence(seq, int_to_vocab, char_tokenization=False):
    """
    Convert a sequence of ids to a sentence
    :param sentence: String
    :param int_to_vocav: Dictionary to go from the ids to words
    :return: the sentence string
    """
    if char_tokenization:
        return ''.join(word for word in [int_to_vocab[i] for i in seq] if word != '<PAD>'and word != '<EOS>')
    else:
        return ' '.join(word for word in [int_to_vocab[i] for i in seq] if word != '<PAD>'and word != '<EOS>')

def sentence_to_seq(sentence, vocab_to_int):
    """
    Convert a sentence to a sequence of ids
    :param sentence: String
    :param vocab_to_int: Dictionary to go from the words to an id
    :return: List of word ids
    """
    sentence = sentence.lower()
    words = sentence.split(" ")
    return [vocab_to_int[word] if word in vocab_to_int.keys() else vocab_to_int['<UNK>'] for word in words ]

def ints_to_sentence(ints, target_int_to_vocab):
    words = [target_int_to_vocab[j] for j in ints]
    sentence = ' '.join(word for word in words if word != '<PAD>'and word != '<EOS>')
    return sentence

def get_avg_reward(hypotheses, references, sources, target_int_to_vocab, source_int_to_vocab):
    
    rewards = []
    for ref, hyp, source in zip(references, hypotheses, sources):
        ref_sent = ints_to_sentence(ref, target_int_to_vocab)
        hyp_sent = ints_to_sentence(hyp, target_int_to_vocab)
        source_sent = ints_to_sentence(source, source_int_to_vocab)
        rewards.append(get_reward(ref_sent, hyp_sent, source_sent))

    mean = np.mean(rewards)
    std = np.std(rewards)
    ci_low, ci_high = scipy.stats.norm.interval(0.95,loc=mean,scale=std)

    return mean, std, ci_low, ci_high
    
def get_reward(reference, hypothesis, source):
    #return gleu_calc.sentence_gleu(hypothesis, reference, source)
    try:
        reward = reward_function((reference, hypothesis))
    except:
        return - textdistance.levenshtein(reference, hypothesis)   
        #return - Levenshtein.distance(reference, hypothesis)
    #print("reward",reward)
    return reward
    
    #lev_dist = Levenshtein.distance(reference, hypothesis)
    
    #return - lev_dist

def get_advantages(samples, reference, source, target_int_to_vocab, source_int_to_vocab):
    
    if not isinstance(reference, str):
        reference_words = [target_int_to_vocab[j] for j in reference]
        reference = ' '.join(word for word in reference_words if word != '<PAD>'and word != '<EOS>')
 
    if not isinstance(source, str):
        source_words = [source_int_to_vocab[j] for j in source]
        source = ' '.join(word for word in source_words if word != '<PAD>'and word != '<EOS>')

    rewards = [get_reward(reference, sample, source) for sample in samples]
    avg_reward = sum(rewards)/len(rewards)
    std_dev = np.array(rewards).std()

    # fix for div by zero
    if std_dev == 0:
        advantage_input = np.array([(x-avg_reward) for x in rewards])
    else: 
        advantage_input = np.array([(x-avg_reward)/std_dev for x in rewards])

    return advantage_input, avg_reward

def mask_probs(probs, outputs):
    mask = np.sign(np.abs(outputs)).astype(np.float32)
    return np.multiply(probs, mask)


def filter_duplicate_sentences(sentences):

    unique_sentence_idxs = []
    unique_sentences = []

    for i,s in enumerate(sentences):
            if s not in unique_sentences:
                    unique_sentences.append(s)
                    unique_sentence_idxs.append(i)

    return unique_sentences, unique_sentence_idxs


def reinforce_on_sentence(model, max_length, n_samples, source, reference, learning_rate, keep_prob, sess, target_int_to_vocab, source_int_to_vocab, accum_gradient = True):
    """
    outputs = [model.sample, model.probs, model.loss, model.accum_ops]
    inputs = [model.input_data, model.keep_prob, model.lr, model.advantage]

    setup = sess.partial_run_setup(outputs, inputs)

    # get samples
    samples_out = sess.partial_run(
        setup,
        model.sample,
        {model.input_data: [source]*n_samples,
        model.keep_prob: keep_prob})

    advantages_in, avg_reward = get_advantages(samples_out, reference, source, target_int_to_vocab)

    loss, _, probs_out = sess.partial_run(
        setup,
        [model.loss, model.accum_ops, model.probs],
        { model.advantage: advantages_in,
          model.lr: learning_rate})


    """
    samples_out, probs_out = sess.run(
        [model.sample, model.probs],
        {model.input_data: [source]*n_samples,
        model.keep_prob: keep_prob,
        model.lr: learning_rate,
        model.max_sample_length: max_length})

    sample_sentences = [seq_to_sentence(sen, target_int_to_vocab) for sen in samples_out]
    unique_samples, unique_sample_idxs = filter_duplicate_sentences(sample_sentences)

    unique_sample_mask = [ 0 for i in range(len(sample_sentences))]
    for idx in unique_sample_idxs:
        unique_sample_mask[idx] = 1
    
  
    advantages_input, avg_reward = get_advantages(unique_samples, reference, source, target_int_to_vocab, source_int_to_vocab)

    if accum_gradient:
        train_op = model.accum_ops
    else: 
        train_op = model.slow_train_op

    loss, _ = sess.run(
        [model.loss, train_op],
        {model.input_data: [source]*n_samples,
         model.chosen_ids: samples_out,
         model.advantage: advantages_input,
        model.keep_prob: keep_prob,
        model.lr: learning_rate,
        model.unique_sample_mask: unique_sample_mask,
        model.max_sample_length: max_length})

    # mask probs and get total prob per sample
    masked_probs = mask_probs(probs_out, samples_out)
    total_probs = np.sum(masked_probs, axis=-1)

    return avg_reward, loss, samples_out, total_probs


def get_n_most_likely_samples(n, samples, probs):
    top_idxs = probs.argsort()[-n:][::-1]
    return samples[top_idxs], probs[top_idxs]

def eval_validation_set(valid_source, valid_target, batch_size, model, sess, target_int_to_vocab, source_int_to_vocab, limit = None):

    if limit is not None:
        valid_target = valid_target[:limit]
        valid_source = valid_source[:limit]

    # sort by lengths to minimize padding
    # this helps reduce differences across batch sizes
    #valid_source, valid_target = zip(*sorted(zip(valid_source, valid_target), key=lambda x: len(x[0]), reverse=True))

    accs = []
    batch_sizes = []
    outputs = []

    max_target_len = max([len(x) for x in valid_target])

    # evaluate batch wise
    for val_batch_start_i in range(0, len(valid_source), batch_size):
        print(val_batch_start_i, batch_size)
        val_target_batch = valid_target[val_batch_start_i:val_batch_start_i + batch_size]
        val_source_batch = valid_source[val_batch_start_i:val_batch_start_i + batch_size]
        
        # pad batches for sentence length
        val_target_batch = np.array(helper.pad_sentence_batch(val_target_batch))
        val_source_batch = np.array(helper.pad_sentence_batch(val_source_batch))

        # pad batches if not enough examples
        if len(val_source_batch) < batch_size:
            diff = batch_size - len(val_source_batch)
            source_padding = np.array([[0]*len(val_source_batch[0])]*diff)
            target_padding = np.array([[0]*len(val_target_batch[0])]*diff)
           
            val_target_batch_pad = np.concatenate((val_target_batch,target_padding),axis=0)
            val_source_batch_pad = np.concatenate((val_source_batch,source_padding),axis=0)

            logits, accuracy = sess.run([model.inference_logits, model.inference_accuracy],
                    {model.input_data: val_source_batch_pad, model.targets: val_target_batch_pad, model.keep_prob: 1.0, model.sequence_length: val_target_batch_pad.shape[1]-1})

            # only pick relevant logits
            logits = logits[:len(val_source_batch)]
        else:
            logits, accuracy = sess.run([model.inference_logits, model.inference_accuracy],
                    {model.input_data: val_source_batch, model.targets: val_target_batch, model.keep_prob: 1.0, model.sequence_length: val_target_batch.shape[1]-1})
        
        batch_sizes.append(len(logits))
        accs.append(accuracy)
        out = np.argmax(logits, 2)
        out = np.pad(out, [(0, 0), (0, max_target_len - len(out[0]))], mode='constant')
        outputs.append(out)
    
    preds = np.concatenate(outputs)    
    avg_reward, std, ci_low, ci_high = get_avg_reward(preds, valid_target, valid_source, target_int_to_vocab, source_int_to_vocab)

    # do weighted average
    weights = [x / len(valid_source) for x in batch_sizes]
    weighted_acc = sum([x*w for x,w in zip(accs, weights)])
 
    return float(weighted_acc), avg_reward, std, ci_low, ci_high, preds

def log_eval(line, file): 
    with open(file, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(line)

def eval_full(train_writer, eval_model, eval_batch_size, step, sess, valid_source, valid_target, target_int_to_vocab, source_int_to_vocab, log_file, save = True):

    acc, avg_reward, reward_std, reward_ci_low, reward_ci_high, _ = eval_validation_set(valid_source, valid_target, eval_batch_size, eval_model, sess, target_int_to_vocab, source_int_to_vocab)

    summary = sess.run(
        eval_model.merged_epoch_summaries,
        {   eval_model.avg_full_dev_accuracy: acc,
            eval_model.avg_full_dev_reward: avg_reward,
        }
    )

    if save: 
        train_writer.add_summary(summary, step)
        log_eval([step, acc, avg_reward, reward_std, reward_ci_low, reward_ci_high], log_file)
        print("acc: ", str(acc), "reward: ", str(avg_reward))
    return acc

def train(experiment_name, method, model, epochs, input_batch_size, train_source, train_target, valid_source, valid_target, learning_rate, keep_probability, save_path, start_checkpoint, target_int_to_vocab, source_int_to_vocab, source_vocab_to_int, log_dir, graph_batch_size, max_hours, eval_model, eval_batch_size, reward_func, early_stopping = False):

    print("training...")

    # FIX
    global reward_function
    reward_function = reward_func

    tf.set_random_seed(1234)

    small_valid_source = np.array(helper.pad_sentence_batch(valid_source[:graph_batch_size]))
    small_valid_target = np.array(helper.pad_sentence_batch(valid_target[:graph_batch_size]))

    small_eval_size = 100
    if method == 'reinforce':
        small_eval_frequency = 10
        full_eval_frequency = 100
    else:
        small_eval_frequency = 1000
        full_eval_frequency = 'epoch'

    save_every_x_epoch = 1
    global_step = 0

    saver = tf.train.Saver(max_to_keep=None)
    train_writer = tf.summary.FileWriter(log_dir)

    log_file = log_dir + '/dev_eval.csv'
    log_eval(['step', 'acc', 'reward', 'reward_std', 'ci_low', 'ci_high'], log_file) # log header
    start_time = time.time()

    best_valid_acc = 0.

    with tf.Session() as sess:

        if start_checkpoint is not None:
            saver.restore(sess, start_checkpoint)
            sess.run(tf.local_variables_initializer())
        else:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
        
        for epoch_i in range(epochs):

            if max_hours is not None:                    
                    hours, rem = divmod(time.time()-start_time, 3600)
                    if hours >= max_hours:
                        break
            
            if full_eval_frequency == 'epoch': # evaluate full dev set
                eval_full(train_writer, eval_model, eval_batch_size, epoch_i, sess, valid_source, valid_target, target_int_to_vocab, source_int_to_vocab, log_file)
         
            for batch_i, (source_batch, target_batch) in enumerate(helper.batch_data(train_source, train_target, input_batch_size)):
                
                if full_eval_frequency != 'epoch' and batch_i % full_eval_frequency == 0:
                    eval_full(train_writer, eval_model, eval_batch_size, global_step, sess, valid_source, valid_target, target_int_to_vocab, source_int_to_vocab, log_file)
                    saver.save(sess, save_path=save_path, global_step=global_step)
       
                if max_hours is not None:                    
                    hours, rem = divmod(time.time()-start_time, 3600)
                    if hours >= max_hours:
                        break

                if method == 'MLE':
                    _, loss = sess.run(
                        [model.mle_train_op, model.mle_cost],
                        {model.input_data: source_batch,
                         model.targets: target_batch,
                         model.lr: learning_rate,
                         model.sequence_length: target_batch.shape[1],
                         model.keep_prob: keep_probability})

                    probs = [] # fix for logging

                elif method == 'reinforce':
                    # initialize the accumulated gradients holder
                    sess.run(model.zero_ops)

                    for source, target in zip(source_batch, target_batch):
                        # add gradient for sentence
                        avg_reward, loss, samples, probs = reinforce_on_sentence(model, len(target), graph_batch_size, source, target, learning_rate, keep_probability, sess, target_int_to_vocab, source_int_to_vocab)
                    # update based on accumlated gradients
                    sess.run(model.update_batch, {model.lr: learning_rate})


                if batch_i % small_eval_frequency == 0: 
            
                    valid_acc, avg_valid_reward, reward_std, reward_ci_low, reward_ci_high, _ = eval_validation_set(valid_source, valid_target, eval_batch_size, eval_model, sess, target_int_to_vocab, source_int_to_vocab, small_eval_size)

                    curr_time = time.strftime("%H:%M:%S", time.localtime())
                    output = experiment_name + ': Epoch {:>3} Batch {:>4}/{} - Validation Accuracy: {:>6.3f}, Loss: {:>6.3f}, Val Reward: {:>6.3f}, time: {}'.format(epoch_i, batch_i, len(train_source) // input_batch_size, valid_acc, loss, avg_valid_reward, curr_time)
                    print(output)

                    """
                    translate_sentence = 'safety is one of the crucial problem that many country and companies concern .'
                    translate_sentence = sentence_to_seq(translate_sentence, source_vocab_to_int)
                    translate_logits = sess.run(model.inference_logits, {model.input_data: [translate_sentence]*graph_batch_size, model.keep_prob: 1.0, model.sequence_length: len(translate_sentence)})[0]
                    print(np.argmax(translate_logits, 1))
                    print(' '.join(word for word in [target_int_to_vocab[i] for i in np.argmax(translate_logits, 1)] if word != '<PAD>'and word != '<EOS>').encode('utf-8'))
                    """
                    summary = sess.run(
                        eval_model.merged_batch_summaries,
                        {   eval_model.avg_small_dev_accuracy: valid_acc,
                            eval_model.avg_small_dev_reward: avg_valid_reward,
                            eval_model.train_loss: loss,
                            eval_model.probabilities: np.exp(probs),
                            eval_model.log_probabilities: probs
                        }
                    )
                     
                    train_writer.add_summary(summary, global_step)

                global_step += 1

            if method == 'MLE' and early_stopping:
                valid_acc = eval_full(train_writer, eval_model, eval_batch_size, epoch_i, sess, valid_source, valid_target, target_int_to_vocab, source_int_to_vocab, log_file, False)
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    saver.save(sess, save_path=save_path, global_step=epoch_i)
            elif method == 'MLE' and epoch_i % save_every_x_epoch == 0: 
                saver.save(sess, save_path=save_path, global_step=epoch_i)

        # eval final model  
        if full_eval_frequency == 'epoch':
            step = epochs
        else:
            step = global_step

        valid_acc = eval_full(train_writer, eval_model, eval_batch_size, step, sess, valid_source, valid_target, target_int_to_vocab, source_int_to_vocab, log_file)

        if method == 'MLE' and early_stopping:
            if valid_acc > best_valid_acc:
                saver.save(sess, save_path, global_step = step)
        else:
            # Save Model
            saver.save(sess, save_path, global_step = step)
        print('Model Trained and Saved')
        
def reinforce_test(model, checkpoint, source_vocab_to_int, learning_rate, keep_probability, batch_size, target_int_to_vocab, source_int_to_vocab, valid_source, valid_target):

    on_batch = False

    source_batch = np.array(helper.pad_sentence_batch(valid_source[:batch_size]))
    target_batch = np.array(helper.pad_sentence_batch(valid_target[:batch_size]))
    
    saver = tf.train.Saver()

    with tf.Session() as sess:

        if checkpoint is not None:
            saver.restore(sess, checkpoint)
        else:
            sess.run(tf.global_variables_initializer())
            
        for epoch_i in range(5000):

            if on_batch:
                # initialize the accumulated gradients holder
                sess.run(model.zero_ops)

                for source, target in zip(source_batch, target_batch):
                    # add gradient for sentence
                    avg_reward, loss, samples, probs = reinforce_on_sentence(model, len(target), batch_size, source, target, learning_rate, keep_probability, sess, target_int_to_vocab, source_int_to_vocab)
                # update based on accumlated gradients
                sess.run(model.update_batch, {model.lr: learning_rate})

                batch_logits = sess.run(
                    model.inference_logits,
                    {model.input_data: source_batch, model.keep_prob: 1.0, model.sequence_length: target_batch.shape[1]})

                out = np.argmax(batch_logits, 2)

                avg_reward, std, ci_low, ci_high = get_avg_reward(out, target_batch, source_batch, target_int_to_vocab)
                print(avg_reward)
            
            else:
                source_sen = "safety is one of the crucial problem that many country and companies concern ."
                target = "safety is one of the crucial problems that many countries and companies are concerned about ."
                source = sentence_to_seq(source_sen, source_vocab_to_int)
         
                avg_reward, loss, samples, probability = reinforce_on_sentence(model, 16, batch_size, source, target, learning_rate, keep_probability, sess, target_int_to_vocab, source_int_to_vocab, accum_gradient = False)
                
                if epoch_i % 1 == 0: 
                    print("avg reward", avg_reward)
                    best_samples, best_probs = get_n_most_likely_samples(1, samples, probability)

                    for s, p in zip(best_samples, best_probs):
                        hypothesis = seq_to_sentence(s, target_int_to_vocab)
                        print(10**p, get_reward(target, hypothesis, source_sen), hypothesis)
