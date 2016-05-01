import tensorflow as tf
import numpy as np


class cnnlda_class(object):
    def __init__(
      self, embedding_mat, non_static, train_size, sequence_length, num_classes, 
      vocab_size, embedding_size, filter_sizes, num_filters):

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
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Calculate loss
        self.centroid = []
        for i in range(num_classes):
            self.centroid.append(tf.Variable(tf.truncated_normal([1, num_filters_total], stddev= 5)))

        with tf.name_scope("loss"):
            def mean_var(x, mask, mean, num):
                x_mask = tf.mul(x, mask)
                residual = x_mask - mean
                res_mask = tf.mul(residual ,mask)
                res_mask_sq = tf.mul(res_mask, res_mask)
                var = tf.reduce_sum(res_mask_sq,0,keep_dims=True)*1.0/(num+1e-7)
                return tf.reduce_sum(var)

            var_loss = tf.constant(0.0)
            mean_list = []
            nums = tf.reduce_sum(self.input_y,0)
            for i in range(num_classes):
                mask_i = tf.expand_dims(self.input_y[:,i],-1)
                var_i = mean_var(self.h_pool_flat, mask_i, self.centroid[i], nums[i])
                var_loss += var_i

            mean_mat = tf.concat(0, self.centroid)
            mask = tf.ones([num_classes,1])
            var = mean_var(mean_mat, mask, tf.reduce_mean(mean_mat,0,keep_dims=True), num_classes)
            var_loss /= var
            self.loss = var_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            def distance(feature, centroid):
                dist_list = []
                for i in range(num_classes):
                    mean = centroid[i]
                    residual = feature-mean
                    res_sq = tf.mul(residual, residual)
                    dist = tf.reduce_sum(res_sq, 1, keep_dims = True)
                    dist_list.append(dist)
                return tf.concat(1, dist_list)

            dist_all = distance(self.h_pool_flat, self.centroid)
            self.predictions = tf.argmin(dist_all,1)
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
