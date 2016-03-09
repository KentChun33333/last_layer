#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from cnnsvm import cnnsvm_class
from sklearn import svm

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability")
tf.flags.DEFINE_boolean("soft_margin", False, "Use Linear Support Vector Machine with soft margin")
tf.flags.DEFINE_float("C", 30, "Relative weight of W's l2 norm : hinge loss")
tf.flags.DEFINE_float("alpha", 0.2, "Relative weight of constraint penalty on slack vars : hinge loss, use only when margin is soft")
tf.flags.DEFINE_boolean("non_static", False, "Refine pre-trained embeddings during training")

# Training parameters
tf.flags.DEFINE_integer("hold_out", 300, "Default: 600")
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_every", 1, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps")

# Data parameters
tf.flags.DEFINE_boolean("vn", False, "Use Vietnamese dataset")
tf.flags.DEFINE_string("en_embeddings", "GoogleNews-vectors-negative300.bin", "English pre-trained words file name")
tf.flags.DEFINE_string("vn_embeddings", "vectors-phrase.bin.vn", "Vietnamese pre-trained words file name")

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
x_, y_, vocabulary, vocabulary_inv, test_size = data_helpers.load_data(FLAGS.vn)

print("Loading pre-trained vectors...")
trained_vecs = data_helpers.load_trained_vecs(
    FLAGS.vn, FLAGS.vn_embeddings, FLAGS.en_embeddings, vocabulary)

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
        cnn = cnnsvm_class(
            embedding_mat = embedding_mat,
            non_static = FLAGS.non_static,
            train_size = x_train.shape[0],
            sequence_length=x_train.shape[1],
            num_classes=y.shape[1],
            vocab_size=len(vocabulary),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=map(int, FLAGS.filter_sizes.split(",")),
            num_filters=FLAGS.num_filters,
            soft_margin=float(FLAGS.soft_margin),
            C=FLAGS.C,
            alpha=FLAGS.alpha)
        
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
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
        run_folder = 'cnnsvm_run' + int(FLAGS.vn)*'_vn'
        out_dir = os.path.abspath(os.path.join(os.path.curdir, run_folder, timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary])#, grad_summaries_merged])
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

        def train_step(x_batch, y_batch, ind_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
              cnn.pad: np.zeros([FLAGS.batch_size, 1, FLAGS.embedding_dim, 1]),
              cnn.ind: ind_batch
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            print("TRAIN step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0,
              cnn.pad: np.zeros([len(x_batch), 1, FLAGS.embedding_dim, 1]),
              cnn.ind: range(len(x_batch)) # doesn't matter
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            print("VALID step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
            return accuracy, loss

        # Generate batches
        batches = data_helpers.batch_iter(
            zip(x_train, y_train, range(x_train.shape[0])), FLAGS.batch_size, FLAGS.num_epochs)

        # Training loop. For each batch...
        max_acc = 0
        best_at_step = 0
        for batch in batches:
            x_batch, y_batch, ind_batch = zip(*batch)
            train_step(x_batch, y_batch, ind_batch)
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