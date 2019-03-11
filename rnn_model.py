import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.contrib.rnn import LSTMStateTuple

# def length(sequence):
#   used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
#   length = tf.reduce_sum(used, 1)
#   length = tf.cast(length, tf.int32)
#   return length

def length(sequence):
  used = tf.sign(sequence)
  length = tf.reduce_sum(used, 1)
  length = tf.cast(length, tf.int32)
  return length

def model_inputs(batch_size):
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate, keep probability)
    """
    input = tf.placeholder(tf.int32, shape=(batch_size,None), name='input')
    targets = tf.placeholder(tf.int32, shape=(batch_size,None), name='targets')
    
    learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, shape=(), name='keep_prob')
    # TODO: Implement Function
    return input, targets, learning_rate, keep_prob

def process_decoding_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for decoding
    :param target_data: Target Placeholder
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param batch_size: Batch Size
    :return: Preprocessed target data
    """

    # remove the last word id from each batch
    target_data = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])

    # concat go id to beginning of each batch
    GO_id = target_vocab_to_int['<GO>']    
    GO_id_col = tf.fill([batch_size, 1], GO_id)
    target_data = tf.concat([GO_id_col, target_data], axis=1)
        
    return target_data

def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, input_sequence_length):
    """
    Create encoding layer
    :param rnn_inputs: Inputs for the RNN
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param keep_prob: Dropout keep probability
    :return: RNN state
    """

    # bidir encoder inspired from
    # https://github.com/ematvey/tensorflow-seq2seq-tutorials/blob/f767fd66d940d7852e164731cc774de1f6c35437/model_new.py

    #lstm = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_size)
    
    #lstm = tf.contrib.rnn.DropoutWrapper(lstm, keep_prob)
    
    #cell = tf.contrib.rnn.MultiRNNCell([lstm] * num_layers)

    stacked_cells = []
    for _ in range(num_layers):
        lstm = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_size)
        lstm = tf.contrib.rnn.DropoutWrapper(lstm, keep_prob)
        stacked_cells.append(lstm)
    cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_cells)

    ((encoder_fw_outputs, encoder_bw_outputs),(encoder_fw_state, encoder_bw_state)) = tf.nn.bidirectional_dynamic_rnn(
               cell_fw = cell,
               cell_bw = cell,
               sequence_length = input_sequence_length,#length(rnn_inputs),
               inputs = rnn_inputs, 
               dtype=tf.float32)

    #return encoder_bw_outputs, encoder_bw_state
    

    encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
    

    # concat forward and backward encoder states at each layer
    encoded_state = []
    for i in range(len(encoder_fw_state)):
        if isinstance(encoder_fw_state[i], LSTMStateTuple):
            encoder_state_c = tf.concat(
                (encoder_fw_state[i].c, encoder_bw_state[i].c), 1, name='bidirectional_concat_c')
            encoder_state_h = tf.concat(
                (encoder_fw_state[i].h, encoder_bw_state[i].h), 1, name='bidirectional_concat_h')
            encoder_state = LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
            encoded_state.append(encoder_state)

        elif isinstance(encoder_fw_state, tf.Tensor):
            encoder_state = tf.concat((encoder_fw_state[i], encoder_bw_state[i]), 1, name='bidirectional_concat')
            encoded_state.append(encoder_state)

    encoded_state = tuple(encoded_state)
    

    """
    #print("bidir",state)
    #outputs, state = tf.nn.dynamic_rnn(cell, rnn_inputs, dtype=tf.float32)
    #print("unidir",state)
    print(outputs)
    state = tf.concat(state[0], axis=2)
    #print(state[0])
    outputs = tf.concat(outputs, axis=2)
    #state = tf.concat(state, axis=2)
    print("-"*10)
    print(outputs)
    print(state)
    exit(0)
    """
    return encoder_outputs, encoded_state

def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope,
                         output_fn, keep_prob, encoder_outputs):
    """
    Create a decoding layer for training
    :param encoder_state: Encoder State
    :param dec_cell: Decoder RNN Cell
    :param dec_embed_input: Decoder embedded input
    :param sequence_length: Sequence Length
    :param decoding_scope: TenorFlow Variable Scope for decoding
    :param output_fn: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: Train Logits
    """ 

    # attention_states: size [batch_size, max_time, num_units]
    #attention_states = array_ops.transpose(encoder_outputs, [1, 0, 2])
    attention_states = encoder_outputs

    dec_hidden_size = dec_cell.state_size[0].c
    # Prepare attention
    att_keys, att_values, att_score_fn, att_construct_fn = tf.contrib.seq2seq.prepare_attention(
               attention_states, 'bahdanau', dec_hidden_size
               )
    
    decoder_fn_train = tf.contrib.seq2seq.attention_decoder_fn_train(
              encoder_state=encoder_state,
              attention_keys=att_keys,
              attention_values=att_values,
              attention_score_fn=att_score_fn,
              attention_construct_fn=att_construct_fn)

    #decoder_fn_train = tf.contrib.seq2seq.simple_decoder_fn_train(encoder_state)
     
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, keep_prob)
    
    outputs, final_state, final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(
        dec_cell, 
        decoder_fn_train,
        inputs = dec_embed_input,
        sequence_length=sequence_length,
        scope=decoding_scope
        )

    # TODO: Implement Function
    return output_fn(outputs), final_state#final_context_state#final_state


def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id,
                         maximum_length, vocab_size, decoding_scope, output_fn, keep_prob, encoder_outputs):
    """
    Create a decoding layer for inference
    :param encoder_state: Encoder state
    :param dec_cell: Decoder RNN Cell
    :param dec_embeddings: Decoder embeddings
    :param start_of_sequence_id: GO ID
    :param end_of_sequence_id: EOS Id
    :param maximum_length: The maximum allowed time steps to decode
    :param vocab_size: Size of vocabulary
    :param decoding_scope: TensorFlow Variable Scope for decoding
    :param output_fn: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: Inference Logits
    """
    """
    infer_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(output_fn, encoder_state,
    dec_embeddings, start_of_sequence_id, end_of_sequence_id, maximum_length, vocab_size)
    drop = tf.contrib.rnn.DropoutWrapper(dec_cell, keep_prob)
    outputs, state, context = tf.contrib.seq2seq.dynamic_rnn_decoder(drop, infer_fn,
        sequence_length = maximum_length, scope = decoding_scope)
    inference_logits =  outputs
    return inference_logits
    """

        # attention_states: size [batch_size, max_time, num_units]
    #attention_states = array_ops.transpose(encoder_outputs, [1, 0, 2])
    attention_states = encoder_outputs
    
    dec_hidden_size = dec_cell.state_size[0].c
    # Prepare attention
    att_keys, att_values, att_score_fn, att_construct_fn = tf.contrib.seq2seq.prepare_attention(
               attention_states, 'bahdanau', dec_hidden_size
               )
    
    decoder_fn_inference = tf.contrib.seq2seq.attention_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=encoder_state,
                    attention_keys=att_keys,
                    attention_values=att_values,
                    attention_score_fn=att_score_fn,
                    attention_construct_fn=att_construct_fn,
                    embeddings=dec_embeddings,
                    start_of_sequence_id=start_of_sequence_id,
                    end_of_sequence_id=end_of_sequence_id,
                    maximum_length=maximum_length,
                    num_decoder_symbols=vocab_size
                  )

    """
    decoder_fn_inference = tf.contrib.seq2seq.simple_decoder_fn_inference(
        output_fn,
        encoder_state,
        dec_embeddings,
        start_of_sequence_id,
        end_of_sequence_id,
        maximum_length,
        vocab_size)
    """
    
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, keep_prob)
    
    outputs, final_state, final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(
        dec_cell,
        decoder_fn_inference,
        sequence_length=maximum_length,
        scope=decoding_scope)
    
    # TODO: should i apply output_fn?
    return outputs, final_state


def sample(encoder_state, dec_cell, dec_embed_input, max_sample_length, decoding_scope, output_fn, keep_prob, dec_embeddings, batch_size, target_vocab_to_int, encoder_outputs, chosen_ids = None):
    
    sequence_length = tf.ones(shape=(batch_size), dtype=tf.int32)

    def step(i, curr_state, curr_input, outputs, probs):
    
        logits, next_state = decoding_layer_train(curr_state, dec_cell, curr_input, sequence_length, decoding_scope, output_fn, keep_prob, encoder_outputs)
        
        dim = tf.reduce_prod(tf.shape(logits)[:2]) # remove max_len axis (2nd. axis), since it's 1
        logits_reshaped = tf.reshape(logits, [dim, -1])  
        logits_reshaped.set_shape([batch_size, None])

        if chosen_ids is None: # pick output from logits
            # pick from top k
            k = 10
            top_k_logits, top_k_idxs = tf.nn.top_k(logits_reshaped, k, sorted=False)
            top_k_multinomial = tf.to_int32(tf.multinomial(top_k_logits,1, seed=1))
            sample_idxs = tf.range(tf.shape(top_k_idxs)[0])
            sample_idxs = tf.expand_dims(sample_idxs, 1)
            gather_idxs = tf.concat(( sample_idxs, top_k_multinomial), axis=1)
            output_ids = tf.gather_nd(top_k_idxs, gather_idxs) 
            output_ids = tf.expand_dims(output_ids, 1)
            
        else: # pick pre-chosen ids, in order to get their probs
            #output_ids = chosen_ids[:,i]
            output_ids = tf.gather(tf.transpose(chosen_ids), i )        
            output_ids = tf.expand_dims(output_ids, 1)
            
        
        # get prob of output
        softmax = tf.nn.log_softmax(logits_reshaped)  
        softmax_idxs = tf.range(tf.shape(softmax)[0])
        softmax_idxs = tf.expand_dims(softmax_idxs, 1) # match output_ids dimesions
        gather_idxs = tf.concat(( softmax_idxs, output_ids), axis=1)
        prob = tf.gather_nd(softmax, gather_idxs) 

        probs = probs.write(i, prob)
        outputs = outputs.write(i, tf.squeeze(output_ids))
    
        # update input
        next_input = tf.nn.embedding_lookup(dec_embeddings, output_ids)

        i = tf.add(i, 1)

        return i, next_state, next_input, outputs, probs


    i = tf.constant(0)
    probs = tf.TensorArray(dtype=tf.float32, size=max_sample_length)
    outputs = tf.TensorArray(dtype=tf.int32, size=max_sample_length)

    # first input
    dec_input = tf.constant([[target_vocab_to_int['<GO>']]])
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    curr_input = tf.tile(dec_embed_input, [batch_size,1,1]) # replicate input to batch size

    # first state
    curr_state = encoder_state

    condition = lambda i, curr_state, curr_input, outputs, probs: tf.less(i, max_sample_length)
    body = lambda i, curr_state, curr_input, outputs, probs: step(i, curr_state, curr_input, outputs, probs)    
    _, _, _, final_outputs, final_probs = tf.while_loop(condition, body, [i, curr_state, curr_input, outputs, probs])

    outputs = tf.transpose(final_outputs.stack())
    probs = tf.transpose(final_probs.stack())

    return outputs, probs



    """
    sequence_length= tf.ones(shape=(batch_size), dtype=tf.int32)
    max_output_length = 10#50#16#50#tf.shape(encoder_outputs)[1] # input length
    
    # first input
    dec_input = tf.constant([[target_vocab_to_int['<GO>']]])
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    curr_input = tf.tile(dec_embed_input, [batch_size,1,1]) # replicate input to batch size

    # first state
    curr_state = encoder_state
    
    probs = []
    outputs = []

    for i in range(max_output_length):
        logits, curr_state = decoding_layer_train(curr_state, dec_cell, curr_input, sequence_length, decoding_scope, output_fn, keep_prob, encoder_outputs)
       
        dim = tf.reduce_prod(tf.shape(logits)[:2]) # remove max_len axis (2nd. axis), since it's 1
        logits_reshaped = tf.reshape(logits, [dim,-1])  
          
        if chosen_ids is None: # pick output from logits
            # pick from top k
            k = 10
            top_k_logits, top_k_idxs = tf.nn.top_k(logits_reshaped, k, sorted=False)
            top_k_multinomial = tf.to_int32(tf.multinomial(top_k_logits,1, seed=1))
            
            sample_idxs = tf.range(tf.shape(top_k_idxs)[0])
            sample_idxs = tf.expand_dims(sample_idxs, 1)
            gather_idxs = tf.concat(( sample_idxs, top_k_multinomial), axis=1)
            output_ids = tf.gather_nd(top_k_idxs, gather_idxs) 
            output_ids = tf.expand_dims(output_ids, 1)
            
        else: # pick pre-chosen ids, in order to get their probs
            output_ids = chosen_ids[:,i]
            output_ids = tf.expand_dims(output_ids, 1)
        
        # get prob of output
        softmax = tf.nn.log_softmax(logits_reshaped)  
        softmax_idxs = tf.range(tf.shape(softmax)[0])
        softmax_idxs = tf.expand_dims(softmax_idxs, 1) # match output_ids dimesions
        gather_idxs = tf.concat(( softmax_idxs, output_ids), axis=1)
        prob = tf.gather_nd(softmax, gather_idxs) 
     
        probs.append(prob)
        
        # update input
        curr_input = tf.nn.embedding_lookup(dec_embeddings, output_ids)

        outputs.append(tf.squeeze(output_ids))
        
    outputs = tf.stack(outputs, axis=1)
    probs = tf.stack(probs, axis=1)

    return outputs, probs
    """

def decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size,
                   num_layers, target_vocab_to_int, keep_prob, batch_size, chosen_ids, encoder_outputs, max_sample_length):
    """
    Create decoding layer
    :param dec_embed_input: Decoder embedded input
    :param dec_embeddings: Decoder embeddings
    :param encoder_state: The encoded state
    :param vocab_size: Size of vocabulary
    :param sequence_length: Sequence Length
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param keep_prob: Dropout keep probability
    :return: Tuple of (Training Logits, Inference Logits)
    """
    # TODO: Implement Function
    
    start_of_sequence_id =  target_vocab_to_int['<GO>']
    end_of_sequence_id = target_vocab_to_int['<EOS>']
    
    #lstm = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_size)
    #lstm = tf.contrib.rnn.DropoutWrapper(lstm, keep_prob)
    #dec_cell = tf.contrib.rnn.MultiRNNCell([lstm] * num_layers)

    stacked_cells = []
    for _ in range(num_layers):
        lstm = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_size)
        lstm = tf.contrib.rnn.DropoutWrapper(lstm, keep_prob)
        stacked_cells.append(lstm)
    dec_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_cells)
    
    output_fn = lambda x: tf.contrib.layers.fully_connected(x, vocab_size, None, scope=decoding_scope)
    
    with tf.variable_scope('decoding') as decoding_scope:
        training_logits, training_final_state = decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope, output_fn, keep_prob, encoder_outputs)

    with tf.variable_scope('decoding', reuse=True) as decoding_scope:
        inference_logits, inference_final_state = decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id, sequence_length, vocab_size, decoding_scope, output_fn, keep_prob, encoder_outputs)
    
    with tf.variable_scope('decoding', reuse=True) as decoding_scope:
        sample_out, probs = sample(encoder_state, dec_cell, dec_embed_input, max_sample_length, decoding_scope, output_fn, keep_prob, dec_embeddings, batch_size, target_vocab_to_int, encoder_outputs)
        
        _, predefined_probs = sample(encoder_state, dec_cell, dec_embed_input, max_sample_length, decoding_scope, output_fn, keep_prob, dec_embeddings, batch_size, target_vocab_to_int, encoder_outputs, chosen_ids)
    return training_logits, inference_logits, sample_out, probs, predefined_probs

def embed_input(input_data, source_vocab_size, enc_embedding_size):
    """
    with tf.name_scope("embeddings"):
        embed = tf.Variable(
                tf.truncated_normal([source_vocab_size, enc_embedding_size], -1, 1))
    return tf.nn.embedding_lookup(embed, input_data)
    """
    return tf.contrib.layers.embed_sequence(input_data, vocab_size=source_vocab_size, embed_dim=enc_embedding_size)

def seq2seq_model(input_data, target_data, keep_prob, batch_size, sequence_length, source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size, rnn_size, num_layers, target_vocab_to_int, chosen_ids, max_sample_length):
    """
    Build the Sequence-to-Sequence part of the neural network
    :param input_data: Input placeholder
    :param target_data: Target placeholder
    :param keep_prob: Dropout keep probability placeholder
    :param batch_size: Batch Size
    :param sequence_length: Sequence Length
    :param source_vocab_size: Source vocabulary size
    :param target_vocab_size: Target vocabulary size
    :param enc_embedding_size: Decoder embedding size
    :param dec_embedding_size: Encoder embedding size
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: Tuple of (Training Logits, Inference Logits)
    """

    input_sequence_length = length(input_data) 
    
    # Apply embedding to the input data for the encoder.
    embedded_input = embed_input(input_data, source_vocab_size, enc_embedding_size)
    
    # Encode the input 
    encoder_outputs, encoded_state = encoding_layer(embedded_input, rnn_size, num_layers, keep_prob, input_sequence_length)#sequence_length)
    #encoded_state_tensor = tf.identity(encoded_state, name='encoded_state')
    
    # Process target data 
    target_data = process_decoding_input(target_data, target_vocab_to_int, batch_size)
    
    # Apply embedding to the target data for the decoder.
    
    #dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, dec_embedding_size]), name='dec_embeddings')  

    from tensorflow.contrib.framework.python.ops import variables
    dec_embeddings = variables.model_variable(
        'embeddings', shape=[target_vocab_size, dec_embedding_size])

    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, target_data)
    
    #temp
    rnn_size *= 2

    # Decode the encoded input
    training_logits, inference_logits, sample, probs, predefined_probs = decoding_layer(
        dec_embed_input,
        dec_embeddings,
        encoded_state,
        target_vocab_size,
        sequence_length,
        rnn_size,
        num_layers,
        target_vocab_to_int,
        keep_prob, 
        batch_size,
        chosen_ids,
        encoder_outputs,
        max_sample_length
    )
    
    return training_logits, inference_logits, sample, probs, predefined_probs

def mask_probs(probs, outputs):
    mask = tf.to_float(tf.sign(tf.abs(outputs)))
    return tf.multiply(probs, mask)
 
def get_reinf_loss(log_probs, advantages):
    
    log_probs = tf.multiply(log_probs, 0.005)
    normalized_log_probs = tf.nn.softmax(log_probs)
    loss = -tf.reduce_sum(tf.multiply(normalized_log_probs,advantages))

    return loss

class RNN:
    
    def __init__(self, batch_size, max_source_sentence_length, source_vocab_to_int, target_vocab_to_int, encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers, trainable=True):

        #tf.reset_default_graph()
        self.max_sample_length = tf.placeholder(tf.int32, shape=None, name='max_sample_length')
        
        
        self.advantage = tf.placeholder(tf.float32, shape=(None), name='advantage')

        self.unique_sample_mask = tf.placeholder(tf.int32, shape=(None), name='unique_sample_mask')
        self.chosen_ids = tf.placeholder(tf.int32, shape=(batch_size,None), name='chosen_ids')
        self.input_data, self.targets, self.lr, self.keep_prob = model_inputs(batch_size)
        input_shape = tf.shape(self.input_data)

        #TODO: sequence length per individual sentence
        self.sequence_length = tf.placeholder_with_default(max_source_sentence_length, None, name='sequence_length')
    
        self.train_logits, self.inference_logits, self.sample, self.probs, self.predefined_probs = seq2seq_model(
            self.input_data, self.targets, self.keep_prob, batch_size, self.sequence_length, len(source_vocab_to_int), len(target_vocab_to_int),
            encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers, target_vocab_to_int, self.chosen_ids, self.max_sample_length)

        tf.identity(self.inference_logits, 'logits')

        if trainable: 
            with tf.name_scope("optimization"):

                optimizer = tf.train.AdamOptimizer(self.lr)
                #optimizer = tf.train.MomentumOptimizer(self.lr, 0.9)

                with tf.name_scope("reinforce"):
                    
                    self.unique_sample_log_probs = tf.dynamic_partition(self.predefined_probs, self.unique_sample_mask, 2)[1]
                    self.unique_samples = tf.dynamic_partition(self.sample, self.unique_sample_mask, 2)[1]

                    
                    masked_probs = mask_probs(self.unique_sample_log_probs,self.unique_samples)
                    per_sample_probs = tf.reduce_sum(masked_probs, axis=-1)

                    log_probs = tf.multiply(per_sample_probs, 0.005)
                    normalized_log_probs = tf.nn.softmax(log_probs)
                    self.loss = -tf.reduce_sum(tf.multiply(normalized_log_probs,self.advantage))

                    self.slow_train_op = optimizer.minimize(self.loss)

                    # get all trainable variables
                    t_vars = tf.trainable_variables()
                    # create a copy of all trainable variables with `0` as initial values
                    self.accum_tvars = [tf.Variable(tf.zeros_like(t_var.initialized_value()),trainable=False) for t_var in t_vars]                                        
                    # create a op to initialize all accums vars
                    self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_tvars]

                    # compute gradients for a batch
                    batch_grads_vars = optimizer.compute_gradients(self.loss, t_vars)
                    # collect the batch gradient into accumulated vars
                    self.accum_ops = [self.accum_tvars[i].assign_add(batch_grad_var[0]) for i, batch_grad_var in enumerate(batch_grads_vars)]

                    acc_gradients = [(self.accum_tvars[i], batch_grad_var[1]) for i, batch_grad_var in enumerate(batch_grads_vars)]
                    capped_acc_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in acc_gradients if grad is not None]
                    # apply accums gradients 
                    self.update_batch = optimizer.apply_gradients(capped_acc_gradients)
      

                with tf.name_scope("mle"):
                    # MLE optimizer
                    self.mle_cost = tf.contrib.seq2seq.sequence_loss(
                        self.train_logits,
                        self.targets,
                        tf.ones([input_shape[0], self.sequence_length]))            

                    # Gradient Clipping
                    gradients = optimizer.compute_gradients(self.mle_cost)
                    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
                    self.mle_train_op = optimizer.apply_gradients(capped_gradients)

        # Inference accuracy
        self.inference_preds = tf.cast(tf.argmax(self.inference_logits, axis=2), tf.int32)

        # inference predictions tensor can be shorter than target tensor.
        # this happens if the decoder meets <EOS> before the max length in all of the batch

        def pad_second_dim(input, desired_size):
            padding = tf.tile([[0]], tf.stack([tf.shape(input)[0], desired_size - tf.shape(input)[1]], 0))
            return tf.concat([input, padding], 1)

        self.padded_inference_preds = pad_second_dim(self.inference_preds, tf.shape(self.targets)[1])

        # mask padded timesteps, so predicting pads is not rewarded
        # mask is 0 if the target and prediction is pad
        accuracy_mask = tf.sequence_mask(
                            length(tf.add(self.targets, self.padded_inference_preds)),
                            dtype=tf.float32
                        )

        correct_preds = tf.cast(tf.equal(self.padded_inference_preds, self.targets), tf.float32)
        masked_correct_preds = tf.multiply(correct_preds, accuracy_mask)
        self.inference_accuracy = tf.divide(tf.reduce_sum(masked_correct_preds), tf.reduce_sum(accuracy_mask))

        # tensorboard
        self.avg_small_dev_accuracy = tf.placeholder(tf.float32, shape=(), name="avg_small_dev_accuracy")
        self.avg_small_dev_reward = tf.placeholder(tf.float32, shape=(), name="avg_small_dev_reward")
        self.avg_train_accuracy = tf.placeholder(tf.float32, shape=(), name="avg_train_accuracy")
        self.avg_train_reward = tf.placeholder(tf.float32, shape=(), name="avg_train_reward")
        self.train_loss = tf.placeholder(tf.float32, shape=(), name="train_loss")
        self.avg_full_dev_accuracy = tf.placeholder(tf.float32, shape=(), name="avg_full_dev_accuracy")
        self.avg_full_dev_reward = tf.placeholder(tf.float32, shape=(), name="avg_full_dev_reward")

        self.probabilities = tf.placeholder(tf.float32, shape=(None), name="probabilities")
        self.log_probabilities = tf.placeholder(tf.float32, shape=(None), name="log_probabilities")
        
        cost_sum = tf.summary.scalar('cost', self.train_loss)
        small_dev_acc_sum = tf.summary.scalar('small_dev_acc', self.avg_small_dev_accuracy)
        small_dev_rew_sum = tf.summary.scalar('small_dev_rew', self.avg_small_dev_reward)
        full_dev_acc_sum = tf.summary.scalar('full_dev_acc', self.avg_full_dev_accuracy)
        full_dev_rew_sum = tf.summary.scalar('full_dev_rew', self.avg_full_dev_reward)

        prob_hist = tf.summary.histogram("probabilities", self.probabilities)
        log_prob_hist = tf.summary.histogram("log_probabilities", self.log_probabilities)
        
        batch_summaries = [cost_sum, small_dev_acc_sum, small_dev_rew_sum, prob_hist, log_prob_hist]
        epoch_summaries = [full_dev_acc_sum, full_dev_rew_sum]

        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        self.merged_epoch_summaries = tf.summary.merge(epoch_summaries)
        self.merged_batch_summaries = tf.summary.merge(batch_summaries)


