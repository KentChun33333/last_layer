import tensorflow as tf
import numpy as np


class cnnsvm_class(object):
    def __init__(
      self, embedding_mat, non_static, train_size, sequence_length, num_classes, 
      vocab_size, embedding_size, filter_sizes, num_filters, soft_margin, C, alpha):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.pad = tf.placeholder(tf.float32, [None, 1, embedding_size, 1], name = "pad")
        self.ind = tf.placeholder(tf.int32, [None], name = 'ind')

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Extend input to a 4D Tensor, because tf.nn.conv2d requires so.
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if not non_static:
                W = tf.constant(embedding_mat, name = "W")
            else:
                W = tf.Variable(embedding_mat, name = "W")
            emb = tf.nn.embedding_lookup(W, self.input_x)
            emb = tf.expand_dims(emb, -1)

        # mask so that the max-pool operator is taken over real_len, not padded len
        pool_mask = tf.to_float(tf.not_equal(self.input_x, 0))
        pool_mask = tf.expand_dims(tf.expand_dims(pool_mask,-1),-1)
        pool_mask = tf.concat(3, [pool_mask] * num_filters)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):

                # Zero paddings so that the convolution output have dimension batch x sequence_length x emb_size x channel
                num_prio = (filter_size-1) // 2
                num_post = (filter_size-1) - num_prio
                pad_prio = tf.concat(1, [self.pad] * num_prio)
                pad_post = tf.concat(1, [self.pad] * num_post)
                emb_pad = tf.concat(1, [pad_prio, emb, pad_post])

                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    emb_pad,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    tf.mul(h, pool_mask),
                    ksize=[1, sequence_length, 1, 1],
                    strides=[1, sequence_length, 1, 1],
                    padding='SAME',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        feature = tf.reshape(self.h_pool, [-1, num_filters_total])
        feature = tf.concat(1, [feature, feature**2])

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            self.W = tf.Variable(tf.truncated_normal([num_filters_total * 2, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(self.W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(feature, self.W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            
            with tf.device('/cpu:0'), tf.name_scope("eps"):
                self.xi = tf.Variable(tf.ones([train_size, num_classes])*1e-3, name = "eps")
                xi_batch = tf.nn.embedding_lookup(self.xi, self.ind) * soft_margin

            t_ovr = 1 - 2.0 * self.input_y # (-1,1) target of one vs rest scheme
            margin = tf.mul(self.scores, t_ovr)
            hinge = tf.reduce_sum(tf.nn.relu(margin + 1.0 - xi_batch))
            xi_loss = tf.reduce_sum(tf.nn.relu(-xi_batch)) * soft_margin

        # Calculate cross-entropy loss
        with tf.name_scope("loss"):
            self.loss = hinge + alpha * xi_loss + C * l2_loss 

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
