import argparse
import numpy as np
import os
from preprocess import imdb_preprocess,text_to_seq_infer
import tensorflow as tf
# Create the parser for getting test_case and the trained model
infer_parser = argparse.ArgumentParser(description='Test case and model path is taken here')

# all the arguments to be taken from comman line 
infer_parser.add_argument('--test_case',
                       type=str,
                       help='the path to the test review file or ignore this option to give custom review')

infer_parser.add_argument('--model_path',
                       default='./s_review_model.h5',
                       help='the path to the saved model for review classification (By default = "./s_review_model.h5")')

infer_parser.add_argument('--tokenizer_path',
                       default='./trained_tokenizer.pkl',
                       help='the path to the saved tokenizer (By default = "./trained_tokenizer.pkl")')

for_infer = infer_parser.parse_args()

MAX_LEN = 200

LABELS = ['negative', 'positive'] # Possible two labels
test_case_path = for_infer.test_case
model_path = for_infer.model_path
tokenizer_path = for_infer.tokenizer_path

print("Command line inputs....")
print("TestCasePath: ",test_case_path)
print("Inference Mode...")
print(" ")
print("Loading and preprocessing of the test_case....\n")
###############################################
# Load test data and preprocess to remove noise#
###############################################
if test_case_path is None:
    review = input("Type your Review here: ")
    test_data = review
else:
    review = open(test_case_path,'r').read()
    clean_data = imdb_preprocess(test_case_path,infer=True)
    test_data = clean_data.remove_noise()
print("Tokenzing and Padding the data....\n")
# Tokenize and Pad Train data 
test_review = text_to_seq_infer(test_data,tokenizer_path,MAX_LEN)

print("Loading the trained sentiment model....")
imdb_model = tf.keras.models.load_model(model_path)
imdb_model.summary()

# Predict the review - postive or negative 
score = imdb_model.predict(test_review)[0][0]
prediction = LABELS[imdb_model.predict_classes(test_review)[0][0]]

print('REVIEW:', review, '\nPREDICTION:', prediction, '\nSCORE: ', score)


