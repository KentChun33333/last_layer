#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from lstm import lstm_class

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_string("lstm_type", "peephole", "either 'gru', 'basic', or 'peephole'")
tf.flags.DEFINE_integer("hidden_unit", 300, "Number of hidden units in lstm cell")
tf.flags.DEFINE_boolean("non_static", False, "Refine pre-trained embeddings during training")

# Training parameters
tf.flags.DEFINE_integer("hold_out", 300, "Default: 300")
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")

# Data parameters
tf.flags.DEFINE_boolean("vn", False, "Use Vietnamese dataset")
tf.flags.DEFINE_string("en_embeddings", "GoogleNews-vectors-negative300.bin", "English pre-trained words file name")
tf.flags.DEFINE_string("vn_embeddings", "vectors-phrase.bin.vn", "Vietnamese pre-trained words file name")
tf.flags.DEFINE_string("dataset","TREC","The problem's dataset")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS
FLAGS.batch_size
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.iteritems()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparatopn
# ==================================================
# Load data
print("Loading data...")
x_, y_, vocabulary, vocabulary_inv, test_size = data_helpers.load_data(FLAGS.vn, FLAGS.dataset)

print("Loading pre-trained vectors...")
trained_vecs = data_helpers.load_trained_vecs(
    FLAGS.vn, FLAGS.vn_embeddings, FLAGS.en_embeddings, vocabulary, FLAGS.dataset)


# Create embedding lookup table
count = data_helpers.add_unknown_words(trained_vecs, vocabulary)
embedding_mat = [trained_vecs[p] for i, p in enumerate(vocabulary_inv)]
embedding_mat = np.array(embedding_mat, dtype = np.float32)

# Randomly shuffle data
x, x_test = x_[:-test_size], x_[-test_size:]
y, y_test = y_[:-test_size], y_[-test_size:]
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
if FLAGS.hold_out == 0:
    x_train = x_shuffled
    y_train = y_shuffled
    x_dev = x_test
    y_dev = y_test
else:
    # Split train/hold-out/test set
    x_train, x_dev = x_shuffled[:-FLAGS.hold_out], x_shuffled[-FLAGS.hold_out:]
    y_train, y_dev = y_shuffled[:-FLAGS.hold_out], y_shuffled[-FLAGS.hold_out:]

print("Vocabulary Size: {:d}".format(len(vocabulary)))
print("Pre-trained words: {:d}".format(count))
print("Train/Hold-out/Test split: {:d}/{:d}/{:d}".format(len(y_train), len(y_dev), len(y_test)))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        lstm = lstm_class(
            embedding_mat = embedding_mat,
            non_static = FLAGS.non_static,
            lstm_type = FLAGS.lstm_type,
            hidden_unit = FLAGS.hidden_unit,
            sequence_length=x_train.shape[1],
            num_classes=y.shape[1],
            vocab_size=len(vocabulary),
            embedding_size=FLAGS.embedding_dim,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.RMSPropOptimizer(1e-3, decay = 0.9)
        grads_and_vars = optimizer.compute_gradients(lstm.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        run_folder = 'lstm_run_' + FLAGS.datset + int(FLAGS.vn)*'_vn'
        out_dir = os.path.abspath(os.path.join(os.path.curdir, run_folder, timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", lstm.loss)
        acc_summary = tf.scalar_summary("accuracy", lstm.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def real_len(xb):
            return [np.argmin(i + [0]) for i in xb]

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              lstm.input_x: x_batch,
              lstm.input_y: y_batch,
              lstm.dropout_keep_prob: FLAGS.dropout_keep_prob,
              lstm.batch_size: FLAGS.batch_size,
              lstm.real_len: real_len(x_batch)
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, lstm.loss, lstm.accuracy],
                feed_dict)
            lstm.W = tf.clip_by_norm(lstm.W, 3)
            time_str = datetime.datetime.now().isoformat()
            print("TRAIN step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              lstm.input_x: x_batch,
              lstm.input_y: y_batch,
              lstm.dropout_keep_prob: 1.0,
              lstm.batch_size: len(x_batch),
              lstm.real_len: real_len(x_batch)
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, lstm.loss, lstm.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("VALID step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
            return accuracy, loss

        # Generate batches
        batches = data_helpers.batch_iter(
            zip(x_train, y_train), FLAGS.batch_size, FLAGS.num_epochs)
        
        max_acc = 0
        best_at_step = 0
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                acc, loss = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                if acc >= max_acc:
                    if acc >= max_acc: max_acc = acc
                    best_at_step = current_step
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            if current_step % FLAGS.checkpoint_every == 0:
                print 'Best of valid = {}, at step {}'.format(max_acc, best_at_step)

        saver.restore(sess, checkpoint_prefix + '-' + str(best_at_step))
        print 'Finish training. On test set:'
        acc, loss = dev_step(x_test, y_test, writer = None)

