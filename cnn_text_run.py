import sys
import os
import time
import datetime
import json

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from cnn_text import TextCNN
from data_loading_cr import data_preprocessing, batch_iterator, process_data

class Configuration(object):
    """ All parameters for CNN model."""
    raw_data_file = "./data/consumer_complaints.csv"
    cleaned_data_file = "./data/cleaned_train_set.csv"
    cleaned_test_file = "./data/cleaned_test_set.csv"
    vocabulary_dir = "./data/vocab.txt"
    test_percentage = 0.2
    cv_percentage = 0.1
    # Hyperparameters
    embedding_size = 64
    filter_sizes = [4, 5, 6]
    num_filters = 64
    dropout_keep_prob = 0.5
    l2_reg_lambda = 0.05
    seq_length = 900

    # cnn training parameters
    batch_size = 64
    num_epochs = 20
    checkpoint_dir =  "./trained/1536365127/checkpoints/" 
    #best_validation = "./trained/checkpoints/best_validation/"
    #word2vec_path = "./data/word2vec.bin"
    use_pretrained_embedding = False
    evaluate_every = 500
    learning_rate = 0.001
    save_per_batch = 10 # save summary every 10 batch

    allow_soft_placement = True
    log_device_placement = False

def train(x_train, y_train, x_cv, y_cv, vocabulary_size, config):
    print("Start trainning cnn model......")
    graph = tf.Graph()
    with graph.as_default():
        session_config = tf.ConfigProto(allow_soft_placement=config.allow_soft_placement, log_device_placement=config.log_device_placement)
        sess = tf.Session(config=session_config)
        with sess.as_default():
            cnn = TextCNN(seq_length=x_train.shape[1], num_classes=y_train.shape[1], vocabulary_size=vocabulary_size, embedding_size=config.embedding_size, filter_sizes=config.filter_sizes, num_filters=config.num_filters, l2_reg_lambda=config.l2_reg_lambda)

            # Define training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(config.learning_rate)
            gradients_and_variables = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(gradients_and_variables, global_step=global_step)

########################### track of gradient values and sparsity (optional)
            """gradient_summaries = []
            for g, v in gradients_and_variables:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    gradient_summaries.append(grad_hist_summary)
                    gradient_summaries.append(sparsity_summary)
            gradient_summaries_merged = tf.summary.merge(gradient_summaries)"""
#################################################################

            # output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "trained", timestamp))
            print("Writing to {}\n".format(out_dir))

            loss_summaries = tf.summary.scalar("loss", cnn.loss)
            accuracy_summaries = tf.summary.scalar("accuracy", cnn.accuracy)

            # train summaries
            train_summary_op = tf.summary.merge([loss_summaries, accuracy_summaries])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # cv summaries
            cv_summary_op = tf.summary.merge([loss_summaries, accuracy_summaries])
            cv_summary_dir = os.path.join(out_dir, "summaries", "cross-validation")
            cv_summary_writer = tf.summary.FileWriter(cv_summary_dir, sess.graph)

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            print("Finish setup summaries!")

            # Initialize Saver
            saver = tf.train.Saver(tf.global_variables())
            # initialize all global variables
            sess.run(tf.global_variables_initializer())

            print("Start training and evaluating......")
            # One training step: train the model with one batch
            def train_step(batch_x, batch_y):
                feeds = {
                    cnn.input_x: batch_x,
                    cnn.input_y: batch_y,
                    cnn.dropout_keep_prob: config.dropout_keep_prob
                }

                _, step, summaries, loss, accuracy = sess.run([train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feeds)

                if (step % config.save_per_batch) == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: Step: {}, Train loss: {}, Train accuracy: {}".format(time_str, step, loss, accuracy))
                    train_summary_writer.add_summary(summaries, step)

            def evaluate_model(x_cv, y_cv, batch_size, summary_op, writer=None):
                eval_batches = batch_iterator(list(zip(x_cv, y_cv)), batch_size, 1)
                total_loss = 0.0
                total_accuracy = 0.0

                for eval_batch in eval_batches:
                    x_batch, y_batch = zip(*eval_batch)
                    batch_len = len(x_batch)
                    feeds = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: 1.0}
                    _, step, summaries, loss, accuracy = sess.run([train_op, global_step, summary_op, cnn.loss, cnn.accuracy], feeds)
                    total_loss += loss * batch_len
                    total_accuracy += accuracy * batch_len
                if writer:
                    writer.add_summary(summaries, step)
                data_len = len(x_cv)
                return total_loss / data_len, total_accuracy / data_len

            # Generate batches
            train_batches = batch_iterator(list(zip(x_train, y_train)), config.batch_size, config.num_epochs)
            best_accuracy = 0 # record best accuracy on validation set
            best_step = 0 # track the step of best accuracy 
            max_improvement_steps = 8000 # if no accuracy improvement between max_improvement_step, stop trainning

            # Training loop: train cnn model with x_train and y_train batch by batch
            for train_batch in train_batches:
                train_batch_x, train_batch_y = zip(*train_batch)
                train_step(train_batch_x, train_batch_y)
                current_step = tf.train.global_step(sess, global_step)

                # validate the model with cross_validation set
                if current_step % config.evaluate_every == 0:
                    print("\nEvaluation on cross validation set...")
                    cv_loss, cv_accuracy = evaluate_model(x_cv, y_cv, config.batch_size, cv_summary_op, cv_summary_writer)
                    print("Loss on cross validation set: {}, accuracy on cross validation set: {}".format(cv_loss, cv_accuracy))
                    # if cv_accuracy is greater than current best_accuracy on cv set at step, save the model
                    if cv_accuracy > best_accuracy:
                        best_accuracy, best_step = cv_accuracy, current_step
                        checkpoint_path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Current best accuracy: {} at step: {}. Save model at {}\n".format(best_accuracy, best_step, checkpoint_path))

                if current_step - best_step > max_improvement_steps:
                    # if there is no improvement over max_improvement_steps, terminate the training
                    print("\nNo optimization for over {}, terminating training...".format(max_improvement_steps))
                    break          
###############################################################################

def test(test_file, vocabulary_dir, checkpoint_dir, seq_length):
    print("Loading test data......")
    start_time = time.time()
    df = pd.read_csv(test_file, index_col=0, encoding="utf-8")
    x_raw = df['text'].tolist()
    y_raw = df['product'].tolist()

    x_test, y_test, _ = process_data(x_raw, y_raw, vocabulary_dir, seq_length);
    print("Test set sample size: {}\n".format(len(x_test)))

    categories = json.loads(open('./data/labels.json').read())
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

    # create session
    graph = tf.Graph()
    with graph.as_default():
        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_config)
        sess.run(tf.global_variables_initializer())
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess=sess, save_path=checkpoint_file)
            print("Loaded the trained model: {}\n".format(checkpoint_file))
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            predictions = graph.get_operation_by_name("scores/predictions").outputs[0]

            print("Testing on test set......")
            test_batches = batch_iterator(list(zip(x_test, y_test)), 128, 1, shuffle=False)
            y_pred = []
            for test_batch in test_batches:
                x_batch, y_batch = zip(*test_batch)
                batch_len = len(x_batch)
                feeds = {input_x: x_batch, dropout_keep_prob: 1.0}
                batch_pred = sess.run(predictions, feeds)
                y_pred = np.concatenate([y_pred, batch_pred])
            
    y_true = np.argmax(y_test, axis=1)
    correct_pred = sum(y_pred == y_true)

    # get the predict labels
    pred_labels = [categories[int(pred)] for pred in y_pred]
    df['prediction'] = pred_labels

    test_accuracy = float(correct_pred) / len(y_true)
    print("Accuracy on test set: {}\n".format(test_accuracy))

    print("Evaluating precision, recall and F1-score on test set......")
    print(metrics.classification_report(y_true, y_pred, target_names=categories))
    json.dump(metrics.classification_report(y_true, y_pred, target_names=categories), open("./data/test_metrics.json", 'w'), indent=4)

    print("Building confusion matrix......")
    confusion = metrics.confusion_matrix(y_true, y_pred)
    json.dump(metrics.classification_report(y_true, y_pred, target_names=categories), open("./data/test_confusion.json", 'w'), indent=2)
    print(confusion)
    print("Time usage: {}\n".format(datetime.timedelta(seconds=int(time.time() - start_time))))

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("Please enter the argument: python cnn_text_train.py [train / test]")

    config = Configuration()
    if sys.argv[1] == 'train':
        x_train, y_train, x_cv, y_cv, vocabulary_size = data_preprocessing(config.raw_data_file, config.cleaned_data_file, config.cleaned_test_file, config.vocabulary_dir, config.test_percentage, config.cv_percentage, config.seq_length)
        train(x_train, y_train, x_cv, y_cv, vocabulary_size, config)
    else:
        test(config.cleaned_test_file, config.vocabulary_dir, config.checkpoint_dir, config.seq_length)



