# Multi-Class-Text-Classification-For-Financial-Consumer-Complaints
## Overview
- In this project, I built a multi-classifier based on convolutional neural network (CNN) using TensorFlow to classify finance consumer complaint narratives into 12 pre-defined categories. 
- Obtained a test accuracy of 86% and average F1 score of 86%.

## Requirements
- Python 3
- Tensorflow > 1.2
- Numpy
- Pandas
- scikit-learn

## Dataset:
[Consumer Complaints Database](https://www.kaggle.com/sebastienverpile/consumercomplaintsdata)
The dataset consists of ~900k examples.

## Data Preprocessing

### Data cleaning:
- Extract consumer complaint narratives and corresponding products from the raw dataset.
- Shuffle the data by rows.
- Cleaned the consumer complaint narratives using regular expression and drop any NA data.
- Divide the cleaned data into training set and test set (default test set percentage: 20%).
- Save the cleaned datasets as csv files to the dir of "./data/".

### Data loading:
- Built a vocabulary index to convert sentences into vectors.
- Padded each sentence into the same length.
- Created label dictionary and converted labels into one hot matrices.
- Randomly split the train set into train set and cross validation set (default cv_percentage: 10%).

## Convolution Neural Network

### Model
- Word embedding layer
- Convolution layer
- Max-pooling layer
- Fully-connected with dropout layer
- Softmax layer

### Configuration
  All the CNN hyperparameters and other configurations can be set in Configuration class in the cnn_text_run.py.

### Train
`python cnn_text_run.py train`

### Test
`python cnn_text_run.py text`

Replace the checkpoint dir in the Configuration class with the output from the training.

`Metrics:`
![alt text][logo1]

[logo1]: https://github.com/xiaojinhe/Multi-Class-Text-Classification-For-Financial-Consumer-Complaints/blob/master/metrics.png "Metrics"


`Confusion Matrix:`
![alt text][logo2]

[logo2]: https://github.com/xiaojinhe/Multi-Class-Text-Classification-For-Financial-Consumer-Complaints/blob/master/confusion.png "Confusion Matrix"

## References
[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

[Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
