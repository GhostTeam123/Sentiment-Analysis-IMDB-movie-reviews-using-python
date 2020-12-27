import argparse
import numpy as np
from preprocess import imdb_preprocess, text_to_seq
from sklearn.model_selection import train_test_split
from model import sentiment_model, plot_acc, plot_loss
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
# Create the parser for getting model config and data path
train_parser = argparse.ArgumentParser(description='Model parameters for training and path to dataset')

# all the arguments to be taken from comman line 
train_parser.add_argument('-p','--path',
                       default='./aclImdb/',
                       help='the path to dataset (By default = "./aclImdb/")')


model_config = train_parser.parse_args()


PATH = model_config.path


# Hyperparams

BATCH_SIZE = 64
EPOCHS = 2
VOCAB_SIZE = 20000
MAX_LEN = 200
EMBEDDING_DIM = 40

print("Command line inputs....")
print("Dataset Path: ",PATH)
print("Training Mode...")
print("Batch_size:",BATCH_SIZE)
print("Number of Epochs:",EPOCHS)
print(" ")
print("Loading and preprocessing of dataset....\n")
###############################################
# Load dataset and preprocess into a csv file #
###############################################
# Train data preprocess
clean_train = imdb_preprocess(in_path=PATH)
train_data = clean_train.populate_csv()
train_data= clean_train.remove_noise()

# Test data preprocess
clean_val = imdb_preprocess(PATH,test=True)
val_data = clean_val.populate_csv()
val_data = clean_val.remove_noise()

print("Tokenzing and Padding the data....\n")
# Tokenize and Pad Train data 
x_train, y_train,x_val,y_val = text_to_seq(train_data,val_data,VOCAB_SIZE, MAX_LEN)

# Convert the data into numpy array with float dtype
X_train = np.asarray(x_train, dtype='float32')
Y_train = np.asarray(y_train, dtype='float32')
X_val = np.asarray(x_val, dtype='float32')
Y_val = np.asarray(y_val, dtype='float32')

print("Creating and Compileing Model pipeline...\n")
#Build model pipeline 
model_imdb = sentiment_model(VOCAB_SIZE,EMBEDDING_DIM,MAX_LEN)

# Initiating the training 
history_imdb = model_imdb.fit(X_train, Y_train, 
          batch_size=BATCH_SIZE, 
          epochs=EPOCHS, 
          validation_split=0.1,
          verbose=2)

# Save the entire model as a SavedModel.
model_imdb.save('s_review_model.h5')

# plotting the accuracy and loss between training and validation data 
# plot_acc(history_imdb,model_imdb)
# plot_loss(history_imdb,model_imdb)

# Print final results of training the model 
train_loss, train_acc = model_imdb.evaluate(X_train,Y_train,verbose=0)
val_loss, val_acc = model_imdb.evaluate(X_val,Y_val,verbose=0)
print("Training Accuracy = {} , Training Loss = {}".format(train_acc,train_loss))
print("Validation Accuracy = {} , Validation Loss = {}".format(val_acc,val_loss))
