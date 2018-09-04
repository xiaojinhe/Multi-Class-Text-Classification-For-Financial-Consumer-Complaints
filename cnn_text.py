import tensorflow as tf
import numpy as np

class TextCNN(object):
    """
    A CNN model for text classification. 
    It is composed of an word embedding layer, a convolutional layer, a max-pooling layer and a softmax layer.
    """
    def __init__(self, seq_length, num_classes, vocabulary_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda):
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.total_num_filters = len(filter_sizes) * num_filters
        self.l2_reg_lambda = l2_reg_lambda
        # tensorflow placeholders for input data, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, self.seq_length], name = 'input_x') # ist dimension is arbitrary batch size
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name = 'input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name = 'dropout_keep_prob')
        self.l2_loss = tf.constant(0.0) # track l2 regularization loss
        self.cnn()

    def cnn(self):
        """ A convolutional neural network model. """
        # Word Embedding layer:
        # maping vocabulary word indices into low-D vectors
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embedding = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0), name="embedding")
            # 1) get embedding of words in sentence
            self.embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x) # [None, seq_length, embedding_size]
            self.expanded_embedding_inputs = tf.expand_dims(self.embedding_inputs, -1) # [None, seq_length, embeding_size, -1]

        # 2) convolutional layer and maxpoll layer for each filter size
        # For each filter, a) create filters, b) convolution, c) apply nonlinearity, d) max pooling
        max_pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("convolution-pooling-%s" % filter_size):
                # create filters
                filter = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter, stddev =0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name='b')
        # conv operation: conv2d, compute a 2D convolution given 4D input and filter.
        # input: [batch size, seq_length, embedding_size, channels] 
        # filter: [filter size, embedding_size, channels, num_filters]
        # conv2d output for each filter: [1, seq_length-filter_size+1, 1, 1] then, * num_filter => [1, seq_length-filter_size+1, 1, num_filters] => then * batch_size => [batch_size, seq_length-filter_size+1, num_filters]
                conv = tf.nn.conv2d(self.expanded_embedding_inputs, W, strides=[1,1,1,1], padding='VALID', name='conv')
                    # apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu') 

                    # max-pooling: 
                    # output shape: [batch_size, 1, 1, num_filters]
                max_pooled = tf.nn.max_pool(h, ksize=[1, self.seq_length - filter_size + 1, 1, 1], strides=[1,1,1,1], padding='VALID', name="pool")
                max_pooled_outputs.append(max_pooled)
        # combine all maxpooling features and flatten the feature vector's # output to [1, None]
        self.h_pooled_features = tf.concat(max_pooled_outputs, 3) # combine three filter sizes
        # shape after flattern: [batch_size, total_num_filters]
        self.h_pooled_features_flat = tf.reshape(self.h_pooled_features, [-1, self.total_num_filters])

        # add dropout to fully connected layer
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pooled_features_flat, keep_prob=self.dropout_keep_prob) 
            
        # Final (unormalized) scores and predictions
        with tf.name_scope("scores"):
            W = tf.get_variable("W", shape=[self.total_num_filters, self.num_classes], initializer=tf.contrib.layers.xavier_initializer())  
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            self.l2_loss += tf.nn.l2_loss(W) + tf.nn.l2_loss(b)

            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits") # shape: [batch_size, num_classes]
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # mean cross-entropy loss
        with tf.name_scope("loss"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy) + self.l2_reg_lambda * self.l2_loss

        # Accuracy 
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"), name="accuracy")
            



