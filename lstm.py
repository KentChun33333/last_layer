import tensorflow as tf
from tensorflow.models.rnn import rnn_cell, rnn
from tensorflow.models.rnn import seq2seq
from tensorflow.models.rnn.ptb import reader
import numpy as np


class lstm_class(object):

    def __init__(
      self, embedding_mat, non_static, lstm_type, hidden_unit, sequence_length, num_classes, vocab_size,
      embedding_size, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.batch_size = tf.placeholder(tf.int32, name = "batch_size")
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.real_len = tf.placeholder(tf.int32, [None], name = "real_len")
        
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Lookup
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if not non_static:
                W = tf.constant(embedding_mat, name = "W")
            else:
                W = tf.Variable(self.embedding_mat, name = "W")
            inputs = tf.nn.embedding_lookup(W, self.input_x)
    
       # LSTM
        if lstm_type == "gru":
            lstm_cell = rnn_cell.GRUCell(num_units = hidden_unit, input_size = embedding_size)
        else:
            if lstm_type == "basic":
                lstm_cell = rnn_cell.BasicLSTMCell(num_units = hidden_unit, input_size = embedding_size)
            else:
                lstm_cell = rnn_cell.LSTMCell(num_units = hidden_unit, input_size = embedding_size, use_peepholes = True)
        lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob = self.dropout_keep_prob)
        
        self._initial_state = lstm_cell.zero_state(self.batch_size, tf.float32)
        inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, sequence_length, inputs)]
        outputs, state = rnn.rnn(lstm_cell, inputs, initial_state=self._initial_state, sequence_length = self.real_len)
        
        # Collect the appropriate last words into variable output (dimension = batch x embedding_size)
        output = outputs[0]
        one = tf.ones([1, hidden_unit], tf.float32)
        with tf.variable_scope("Output"):
            tf.get_variable_scope().reuse_variables()
            for i in range(1,len(outputs)):
                ind = self.real_len < (i+1)
                ind = tf.to_float(ind)
                ind = tf.expand_dims(ind, -1)
                mat = tf.matmul(ind, one)
                output = tf.add(tf.mul(output, mat),tf.mul(outputs[i], 1.0 - mat))
                
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            self.W = tf.Variable(tf.truncated_normal([hidden_unit, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(self.W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(output, self.W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
