# Data handling and manipulation
import pandas as pd
import numpy as np
from tensorflow.keras.utils import get_file
import tarfile
import os
import re
import string
from sklearn.utils import shuffle 
from matplotlib import pyplot as plt
import pickle

# Preprocessing NLP helpers
import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from sklearn.preprocessing import MinMaxScaler
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Word Embedding helpers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class imdb_preprocess(object):
    """
    Preprocessing: Each review is contained in a txt file. preprocessing will get all the contents from
    the txt file - remove unwanted characters, and stopping words finally keep each review in a csv file 
    along with its rating ( 1 or 0)
    Make a csv file 'imbd_train.csv' and 'imbd_test.csv' or 'imbd_test_case.csv'(if infer is True) that
    contains columns 'Text' and 'Rating'.
        - 'Text' column contains each of the review statements.
        - 'Rating' column contains boolean value '1' for postive review and '0' for Negative Review 
    in_path :  Path of train directory / path of the test case (if infer = True)
    out_path : Path to save the csv file.

    """
    def __init__(self,in_path,out_path='./',test=False,infer=False):
        if infer:
            self.test_data = open(in_path,'r').read()
            self.infer = True
        else:
            self.infer = False
            if not os.path.isdir(in_path):
                data_dir = get_file('aclImdb_v1.tar.gz', 
                'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz', 
                cache_subdir = "datasets",
                hash_algorithm = "auto", 
                extract = True, 
                archive_format = "auto")
                my_tar = tarfile.open(data_dir)
                my_tar.extractall('./') # specify which folder to extract to
                my_tar.close()
            
            if test == True:
                self.data_file = 'imdb_test.csv'
                self.neg_path = os.path.join(in_path,'test/neg') # path of negative reviews
                # self.neg_path = in_path + 'test/neg/' # path of negative reviews
                self.pos_path = os.path.join(in_path,'test/pos') # path of positive reviews
                # self.pos_path = in_path + 'test/pos' # path of positive reviews
                if os.path.isfile('imdb_test.csv'):
                    print('The preprocessed file "imdb_test.csv: already present in the current directory!')
                    self.file_exist = True
                else:
                    self.file_exist = False
                    
            else:
                self.data_file = 'imdb_train.csv'
                self.neg_path = os.path.join(in_path,'train/neg') # path of negative reviews
                # self.neg_path = in_path + 'train/neg/' # path of negative reviews
                self.pos_path = os.path.join(in_path,'train/pos') # path of positive reviews
                self.pos_path = in_path + 'train/pos/' # path of positive reviews
                if os.path.isfile('imdb_train.csv'):
                    print('The preprocessed file "imdb_train.csv: already present in the current directory!')
                    self.file_exist = True
                else:
                    self.file_exist = False
            self.out_file = out_path + self.data_file
        
    def populate_csv(self):
        if self.file_exist:
            self.data = pd.read_csv(self.data_file)
            return self.data

        # Create DataFrame Skeleton
        self.data = pd.DataFrame(columns=['review','rating'])

        # Appending postive reviews to the train_data taken from each positive review text files
        for filename in os.listdir(self.pos_path):
            temp_dict = {}
            review = open(os.path.join(self.pos_path,filename),encoding="utf-8").read()
            temp_dict[self.data.columns[0]] = review
            temp_dict[self.data.columns[1]] = 1
            self.data = self.data.append(temp_dict, ignore_index=True)

        # Appending Negative reviews to the train_data taken from each negative review text files
        for filename in os.listdir(self.neg_path):
            temp_dict = {}
            review = open(os.path.join(self.neg_path,filename),encoding="utf-8").read()
            temp_dict[self.data.columns[0]] = review
            temp_dict[self.data.columns[1]] = 0
            self.data = self.data.append(temp_dict, ignore_index=True)

        self.data.to_csv(self.out_file, encoding='utf-8', index=False)
        return self.data

    def remove_unwanted_elements(self,row):
        # Remove regex pattern for html tags
        html_pattern = r'<[^<>]+>'
        review_html = re.sub(html_pattern,' ',row)
        # Remove regex pattern for special characters and whitespaces
        spl_pattern = r'[^a-zA-z0-9\s]'
        review_spl = re.sub(spl_pattern,' ',review_html)
        # Convert to lower cases
        review_lower=review_spl.lower()
        return review_lower

    def lemmatization(self,row):
        temp = WordNetLemmatizer()
        text= ' '.join([temp.lemmatize(word) for word in row.split()])
        return text

    def remove_stopping_words(self,row):
        # Remove stopping words
        stop_words=set(stopwords.words('english'))
        # Some required words that shouldn't be removed for our dataset from stop_words
        required_words = ["not","very"]
        stop_words = [word for word in stop_words if word not in required_words]
        resultwords  = [word for word in row.split() if word not in stop_words]
        final = [ word for word in resultwords if len(word)>2]
        return final


    def remove_noise(self,shuffle_data = True):
        if self.infer:
            return self.test_data
        else:
            self.data['review'] = self.data['review'].apply(self.remove_unwanted_elements)
            self.data['review'] = self.data['review'].apply(self.lemmatization)
            # self.test_data['review'] = self.test_data['review'].apply(self.stemming)
            self.data['review'] = self.data['review'].apply(self.remove_stopping_words)
            if shuffle_data == True:
                self.data = shuffle(self.data)
                return self.data
            return self.data

# For Training 
def text_to_seq(train_data,val_data,num_vocab, max_len):
    # Tokenize each word into a numerical value and set padding 
    tokenizer = Tokenizer(num_words=num_vocab)
    tokenizer.fit_on_texts(train_data.review)
    pkl_filename = './trained_tokenizer.pkl'
    with open(pkl_filename, 'wb') as file:
        pickle.dump(tokenizer, file)
    train_tokens = tokenizer.texts_to_sequences(train_data.review)
    val_tokens = tokenizer.texts_to_sequences(val_data.review)
    x_train = pad_sequences(train_tokens, maxlen=max_len,padding='post')
    x_val = pad_sequences(val_tokens, maxlen=max_len,padding='post')
    y_train = train_data.rating.values
    y_val = val_data.rating.values
    return x_train, y_train, x_val, y_val

# For Inference
def text_to_seq_infer(data,tokenizer_path,max_len):
    pkl_filename = tokenizer_path
    with open(pkl_filename, 'rb') as file:
        tokenizer = pickle.load(file)
    num_tokens = tokenizer.texts_to_sequences([data])
    test_data= pad_sequences(num_tokens, maxlen=max_len,padding='post')
    return test_data