# Multi-Class-Text-Classification-For-Financial-Consumer-Complaints
- Built a multi-classifier based on convolutional neural network (CNN) using TensorFlow to classify ~120k
finance consumer complaint narratives into 12 pre-defined categories.
- Cleaned the data, padded sentences to same length, built a vocabulary index to convert sentences into vectors.
- Developed a model involving word embedding, convolution, max-pooling, and fully-connected with dropout
layers, visualized the training results on TensorBoard with tf.summaries, and evaluated the model by computing precision, recall, F1 score for each class and confusion matrix using sklearn.metrics.
- Achieved a test accuracy of 86% and average F1 score of 86%, after tuning hyperparameters and regularization.
