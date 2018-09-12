import re
import os
import json
import pickle
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tflearn.data_utils import pad_sequences
from data_preprocess_cr import PreprocessingConfig, DataPreprocessing


def build_vocabulary(x_raw, y_raw, vocabulary_dir):
    #build vocabulary: padding each sentence to the same length and mapping words to ids     

    word_to_id = {}
    id_to_word = {}
    word_to_id["<PAD>"] = 0
    id_to_word[0] = "<PAD>"
    word_to_id["<UNK>"] = 1
    id_to_word[1] = "<UNK>"

    label_to_id = {}
    id_to_label = {}

    words_counter = Counter()
    labels_counter = Counter()
    for i in range(len(x_raw)):
        word_list = x_raw[i].strip().split(" ")
        word_list = [w.strip().replace(" ", "") for w in word_list if w != '']
        label_list = [l.strip() for l in y_raw[i] if l != '']
        words_counter.update(word_list)
        labels_counter.update(label_list)

    word_pairs = words_counter.most_common()
    for i, pair in enumerate(word_pairs):
        word, _ = pair
        word_to_id[word] = i + 2
        id_to_word[i + 1] = word

    label_pairs = labels_counter.most_common()
    for i, pair in enumerate(label_pairs):
        label, num = pair
        label = str(label)
        label_to_id[label] = i
        id_to_label[i] = label

    with open(vocabulary_dir, 'ab') as vf:
        pickle.dump((word_to_id, id_to_word, label_to_id, id_to_label), vf)

    return word_to_id, id_to_word, label_to_id, id_to_label 


def read_vocabulary(vocabulary_dir):
    with open(vocabulary_dir, 'rb') as f:
        return pickle.load(f)
         

def labels_to_one_hot(y_raw):
    # creates label dict and convert y to one_hot_matrix
    labels = sorted(list(set(y_raw)))
    one_hot_matrix = np.zeros((len(labels), len(labels)), int)
    np.fill_diagonal(one_hot_matrix, 1)
    label_dict = dict(zip(labels, one_hot_matrix))

    json.dump(labels, open("./data/labels.json", 'w'), indent=4)
    y = []
    for i in range(0, len(y_raw)):
        y.append(label_dict[y_raw[i]])

    return y

def process_data(x_raw, y_raw, vocabulary_dir, seq_length):
    if not os.path.exists(vocabulary_dir):
        print("Building vocabulary......")
        word_to_id, id_to_word, label_to_id, id_to_label = build_vocabulary(x_raw, y_raw, vocabulary_dir)
    else:
        print("Loading vocabulary......")
        word_to_id, id_to_word, label_to_id, id_to_label = read_vocabulary(vocabulary_dir)

    vocabulary_size = len(word_to_id)
    x = []
    for i, text in enumerate(x_raw):
        word_list = text.strip().split(" ")
        word_list = [w.strip().replace(" ", "") for w in word_list if w != '']
        ids = [word_to_id.get(w, 1) for w in word_list]
        x.append(ids)

    x = pad_sequences(x, maxlen=seq_length, value=0.0)
    y = np.array(labels_to_one_hot(y_raw))
    return x, y, vocabulary_size

def data_preprocessing(raw_data_file, cleaned_data_file, cleaned_test_file, vocabulary_dir, test_percentage, cv_percentage, seq_length):
    # step 1: load data and labels
    if not os.path.exists(cleaned_data_file) or not os.path.exists(cleaned_test_file):
        preprocessingConfig = PreprocessingConfig()
        preprocessing = DataPreprocessing(preprocessingConfig, raw_data_file, cleaned_data_file, cleaned_test_file, test_percentage)
        print("\nSaved training set into cleaned data file: {} and test set into cleaned_test_file".format(cleaned_data_file, cleaned_test_file))

    print("Loading data......")
    df = pd.read_csv(cleaned_data_file, index_col=0, encoding="utf-8")
    x_raw = df['text'].tolist()
    y_raw = df['product'].tolist()

    x, y, vocabulary_size = process_data(x_raw, y_raw, vocabulary_dir, seq_length)
    print("Spliting data......")
    # step 2: ssplit the dataset into train, cross-validation, and test sets
    # plit the dataset into training, cross validation, and test sets
    x_train, x_cv, y_train, y_cv = train_test_split(x, y, test_size=cv_percentage, random_state=10)

    del x, y

    print("Vocabulary size: {}".format(vocabulary_size))
    print("Training/cross-validation split: {}/{}".format(len(x_train), len(x_cv)))
    return x_train, y_train, x_cv, y_cv, vocabulary_size
        
def batch_iterator(data, batch_size, num_epochs, shuffle = True):
    """ 
    A iterator is created to iterate the data batch by batch.
    Shuffle the input data at each epoch.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1 

    for epoch in range(num_epochs):
        shuffled_data = data
        if shuffle:
            indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[indices]

        for batch_num in range(num_batches_per_epoch):
            from_index = batch_num * batch_size
            to_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[from_index : to_index]
