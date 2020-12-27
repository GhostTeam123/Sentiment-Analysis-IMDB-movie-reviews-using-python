# Deep learning model helpers
import tensorflow as tf
from tensorflow.keras.layers import Dense ,Conv1D , Embedding, Dropout , Activation, Flatten,GlobalMaxPooling1D
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt

def sentiment_model(VOCAB_SIZE,EMBEDDING_DIM,MAX_LEN):
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN))
    model.add(Conv1D(250,
                    3,
                    activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(250))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def plot_acc(history,model_obj):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
def plot_loss(history,model_obj):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('-- loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()