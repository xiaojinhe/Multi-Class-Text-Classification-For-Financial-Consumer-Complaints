import re
import numpy as np
import pandas as pd

class PreprocessingConfig(object):
    replace_abbrevation_op = True

class DataPreprocessing(object):

    def __init__(self, config, raw_data_file, cleaned_data_file, cleaned_test_file, test_percentage):
        self.config = config
        self.clean_data(raw_data_file, cleaned_data_file, cleaned_test_file, test_percentage)

    def replace_abbreviation(self, str):
        cleaned_str = re.sub(r"\'ve", "have", str) # I've => I 've
        cleaned_str = re.sub(r"\'s", " \'s", cleaned_str) # It's = > it 's
        cleaned_str = re.sub(r"won\'t", "will not", cleaned_str)
        cleaned_str = re.sub(r"n\'t", " not", cleaned_str) # aren't => are not
        cleaned_str = re.sub(r"\'d", " would", cleaned_str) # I'd => I 'd
        cleaned_str = re.sub(r"\'re", " are", cleaned_str) # they're => they 're
        cleaned_str = re.sub(r"\'ll", " will", cleaned_str) # I'll => I 'll
        cleaned_str = re.sub(r"i\'m", "i am", cleaned_str) 
        cleaned_str = re.sub(r",", " , ", cleaned_str)
        cleaned_str = re.sub(r"!", " ! ", cleaned_str)
        cleaned_str = re.sub(r"\(", " \( ", cleaned_str)
        cleaned_str = re.sub(r"\)", " \) ", cleaned_str)
        cleaned_str = re.sub(r"\?", " \? ", cleaned_str)
        cleaned_str = re.sub(r"[^\x00-\x7f]+", "", cleaned_str)
        cleaned_str = re.sub(r"\S*x{2,}\S*", "xxx", cleaned_str)
        cleaned_str = re.sub(r"\s{2,}", " ", cleaned_str) # delete addtional whitespaces
        return cleaned_str

    def clean_text(self, str):
        cleaned_str = re.sub(r"[^A-Za-z(),!?\'\`]", " ", str)
        cleaned_str = str.lower()

        if self.config.replace_abbrevation_op:
            cleaned_str = self.replace_abbreviation(cleaned_str)

        return cleaned_str.strip()

    def clean_data(self, raw_data_file, cleaned_data_file, cleaned_test_file, test_percentage):
        df = pd.read_csv(raw_data_file)
        selected_columns = ['Product', 'Consumer complaint narrative']
        df.drop(list(set(df.columns) - set(selected_columns)), axis=1, inplace=True)
        df.dropna(axis=0, how='any', subset=selected_columns, inplace=True)
        df.reset_index(drop=True, inplace=True)
        # shuffle the data
        shuffled_df = df.reindex(np.random.permutation(df.index))
        shuffled_df.head()

        cleaned_text = []
        for i in range(0, len(shuffled_df)):
            cleaned_text.append(self.clean_text(shuffled_df[selected_columns[1]][i]))
        cleaned_df = pd.DataFrame(cleaned_text, columns=['text'])
        cleaned_df['product'] = shuffled_df[selected_columns[0]]
        cleaned_df.dropna(axis=0, how='any', inplace=True)
        cleaned_df.reset_index(drop=True, inplace=True)
        print(cleaned_df.info())

        #training set
        training_set = cleaned_df.iloc[0:int(len(cleaned_df) * (1 - test_percentage))]
        print(training_set.info())
        training_set.to_csv(cleaned_data_file, encoding="utf-8")
        # testing set
        test_set = cleaned_df.iloc[len(training_set):]
        test_set.to_csv(cleaned_test_file, encoding="utf-8")

        """small_training_set = cleaned_df.iloc[0:3000]
        small_training_set.reset_index(drop=True, inplace=True)
        small_training_set.to_csv("./data/small_training_set.csv", encoding="utf-8")
        small_test_set = cleaned_df.iloc[3000:4000]
        small_test_set.to_csv("./data/small_test_set.csv", encoding="utf-8")"""

if __name__ == '__main__':
    raw_data_file = "./data/consumer_complaints.csv"
    cleaned_data_file = "./data/cleaned_train_set.csv"
    cleaned_test_file = "./data/cleaned_test_set.csv"
    config = PreprocessingConfig()
    preprocessor = DataPreprocessing(config, raw_data_file, cleaned_data_file, cleaned_test_file, 0.1)









