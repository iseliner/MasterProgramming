# -*- coding: utf-8 -*-

#Import the libraries
import gc
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import json
#import pickle

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from numpy.testing import assert_allclose
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
#from sklearn.feature_selection import SelectFromModel
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.feature_extraction.text import CountVectorizer

from keras.utils import np_utils
#from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint
#from keras.preprocessing import sequence

#import gensim
#import nltk
#from nltk.corpus import brown as brown
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
#from gensim.models import FastText

#IMPORTS DATASET AND LABELS
print('Loading data...')

train_label = pd.read_csv('C:/Users/iseliner/Documents/programming/' +
                          '/data/data/labels/train/labels.train.csv')

dataset_path = ('C:/Users/iseliner/Documents/programming/' +
                '/data/data/essays/train/original/')

bigram_essay_path = ('C:/Users/iseliner/Documents/programming/' +
                     'features/ngram/essays/bigram/')
trigram_essay_path = ('C:/Users/iseliner/Documents/programming/' +
                      'features/ngram/essays/trigram/')
#fourgram_essay_path = ('C:/Users/iseliner/Documents/programming/' +
#                       'features/ngram/essays/4gram/')
'''
dataset_path = ('C:/Users/iseliner/Documents/programming/' +
                '/data/data/essays/train/original/')

bigram_essay_path = ('C:/Users/iseliner/Documents/programming/' +
                     'features/char_ngram/essays/bigram/')
trigram_essay_path = ('C:/Users/iseliner/Documents/programming/' +
                      'features/char_ngram/essays/trigram/')
#fourgram_essay_path = ('C:/Users/iseliner/Documents/programming/' +
#                       'features/char_ngram/essays/4gram/')
'''

#           changes labels index to the test_taker_id
train_label = train_label.set_index('test_taker_id')
#           drop the prompts, since we don't use them
train_label = train_label.drop('speech_prompt', axis=1)
train_label = train_label.drop('essay_prompt', axis=1)

vector_len = 200
embed_len = 100

#Function which imports a datasets from a path and puts it into a list
def makeseq(path, listname):
    for file in os.listdir(path):
        read_file = open(path + str(file))
        row = read_file.read().split()
        read_file.close()
        listname.append(row)
        
def makeseqvec(path, listname):
    for file in os.listdir(path):
        read_file = open(path + str(file))
        row = read_file.read()
        read_file.close()
        listname.append(row)

#Makes vectors from the lists        
def makevectors(listname, vectorlist, target_model):
    for essay in listname:
        essay_sen = [np.zeros(embed_len)] * vector_len
        for i,word in enumerate(essay):
            if word in target_model.wv.vocab:
                enc = target_model.wv[word]
                essay_sen[i] = enc
            else:
                target_model.build_vocab([word], update=True)
    vectorlist.append(essay_sen)

#Appends new dataset features, slicing them to fit to the already existing one
def appenddata(target_data, target_df, label_list, length):
    makeseq(target_data, target_df)
    x = 0
    while x < length:
        label_list.append(label_list[x])
        x += 1
    
#Slices elements that are too long and appends the shorter version
def slicefiles(target_df):
    counter = 0
    for essay in target_df:
        if len(essay) > vector_len:
            if len(essay) > vector_len*2:
                new_essay = essay[vector_len:vector_len*2]
            else:
                new_essay = essay[vector_len:len(essay)]
            old_new_sen = essay[0:vector_len]
            label = label_list[counter]
            target_df[counter] = old_new_sen
            target_df.append(new_essay)
            label_list.append(label)
            if len(essay) > vector_len*3:
                new_essay = essay[vector_len*2:vector_len*3]
                target_df.append(new_essay)
                label_list.append(label)
        counter += 1
        
   
def padvectors(target_df):
    X = []
    for seq in target_df:
        sequence = [np.zeros(embed_len)] * vector_len
        for i,word in enumerate(seq):                               
            sequence[i] = word
        X.append(sequence)
    return X


def vectorizedata(target_df, target_model):
    X = []
    for seq in target_df:
        sequence = []
        for word in seq:
            if word in target_model.wv.vocab:
                enc = target_model.wv[word]
                sequence.append(enc)
            else:
                enc = np.array([0]*embed_len)
                sequence.append(enc)
        X.append(sequence)
    return X    

def makelower(target_list):
    for x in range(len(target_list)):
        for y in range(len(target_list[x])):
            if target_list[x][y].isalpha():
                target_list[x][y] = target_list[x][y].lower()
            

#11000 elements, each containing all words in their respective essay
print('Loading labels and data...')
label_list = []
x = 0
while x < len(train_label):
    label = train_label.iat[x,0]
    label_list.append(str(label))
    x += 1

#Creates the dataset (!)
df = []
makeseq(dataset_path, df)
appenddata(bigram_essay_path,df,label_list,11000)
appenddata(trigram_essay_path,df,label_list,11000)
#appenddata(fourgram_essay_path,df,label_list,11000)

#makelower(df)
slicefiles(df)


#sg=1 is skip-gram, cbow otherwise
print('Building Word2Vec...')

model = Word2Vec(sentences=df, size=100, min_count=0, workers=6, window=5,sg=0, compute_loss=True)
word_vectors = model.wv
word_vectors.save('C:/Users/iseliner/Documents/programming/embedding_models/word2vectors_cbow150_max80000_word1_char23.bin')
#vocab_obj = model.wv.vocab.items()
#print(len(vocab_obj))
#model.get_latest_training_loss()


#Use this when the word vector is final, and comment out the model building above
#model = KeyedVectors.load('C:/Users/iseliner/' +
#                          'Documents/programming/embedding_models/word2vectors_cbow_word123.bin')

#model = FastText(df, size=100, min_count=0, workers=5, window=5,sg=0)


#clf = Pipeline([
#  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
#  ('classification', RandomForestClassifier())
#])
#clf.fit(X, y)

        
print('Preparing data for input into the model...')
#Creates the TRAINING INPUT for the model  

#X_train = np.reshape(x, (x.shape))

label_train = pd.DataFrame(label_list)
#Make the label vectors: y_train(11000,11)
y = label_train.values
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
y_train = np_utils.to_categorical(encoded_y)

print('Setting up x_train and y_train...')

df = vectorizedata(df, model)
X_train = padvectors(df)
#X_train = np.array(X_train)

del df
gc.collect()

#select_model = ExtraTreesClassifier()
#select_model.fit(X_train, y_train)
#X_train = select_model.transform(X_train)

X_train = np.array(X_train)
X_train = np.reshape(X_train, (X_train.shape))

#MODEL _________________________________________________________________
print('Creating model...')

#Initalizing the RNN
nn_model = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
nn_model.add(LSTM(68, return_sequences = True, 
                  input_shape = (X_train.shape[1], X_train.shape[2])))
nn_model.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
nn_model.add(LSTM(68, return_sequences = True))
nn_model.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
nn_model.add(LSTM(68, return_sequences = False))
nn_model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
#nn_model.add(Dense(50, activation='softmax'))

# Adding the output layer
nn_model.add(Dense(11, activation='softmax'))

#Should try the RMSPROP as optimizer
# Compiling the RNN
nn_model.compile(loss = 'categorical_crossentropy',
              optimizer = 'rmsprop', 
              metrics=['accuracy'])

filepath = 'C:/Users/iseliner/Documents/programming/saved_models/LSTMmodel_cbow100_80000max_len200_2hidden_68node_w123.h5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, 
                             save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Fitting the RNN to the Training set
print('Fitting data to the model...')
history = nn_model.fit(X_train, y_train, epochs=10, batch_size=90, 
                       callbacks=callbacks_list)
print('Training complete!')




####LOAD MODEL###########################################
'''
from keras.models import load_model

#load the model
loaded_model = load_model('C:/Users/iseliner/Documents/programming/saved_models/LSTMmodel.h5')
assert_allclose(loaded_model.predict(X_train),
                loaded_model.predict(X_train), 1e-5)

#fit the model
filepath = 'C:/Users/iseliner/Documents/programming/saved_models/LSTMmodel_2.h5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, 
                             save_best_only=True, mode='min')
callbacks_list = [checkpoint]
loaded_model.fit(X_train, y_train, epoch=15, batch_size=50, callbacks=callbacks_list)
'''
from keras import backend as K

currentLearningRate = K.get_value(nn_model.optimizer.lr)
plt.plot(range(1,11), history.history['acc'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

#////////////////////////////////////////////////////////////////////////
#TESTING ________________________________________________________________
test_label = pd.read_csv('C:/Users/iseliner/Documents/programming/data/data/labels/dev/labels.dev.csv')
essay_test_path = ('C:/Users/iseliner/Documents/programming/data/data/essays/dev/original/')
speech_test_path = ('C:/Users/iseliner/Documents/programming/data/data/speech_transcriptions/dev/original/')

#           changes labels index to the test_taker_id
test_label = test_label.set_index('test_taker_id')
#           drop the prompts, since we don't use them
test_label = test_label.drop('speech_prompt', axis=1)
test_label = test_label.drop('essay_prompt', axis=1)

def appendtest(target_data, target_df, label_list,length):
    makeseq(target_data, target_df)
    x = 0
    while x < length:
        label_list.append(label_list[x])
        x += 1

#Slices the test files to size
def slicetestfiles(target_df):
    counter = 0
    for essay in target_df:
        if len(essay) > vector_len:
            old_new_sen = essay[0:vector_len]
            target_df[counter] = old_new_sen
        counter += 1

print('Initializing test set labels')
test_label_list = []
x = 0
while x < len(test_label):
    label = test_label.iat[x,0]
    test_label_list.append(str(label))
    x += 1

#11000 elements, each containing all words in the essay
print('Initializing test data')
test_df = []
makeseq(essay_test_path, test_df)
#makeseq(speech_test_path, test_df)
#appendtest(speech_test_path, test_df, test_label_list,1100)
slicetestfiles(test_df)
test_df = vectorizedata(test_df, model)
test_df = padvectors(test_df)

X_test = np.array(test_df)
X_test = np.reshape(X_test, (X_test.shape))

print('Setting final labels')
label_test = pd.DataFrame(test_label_list)
y = label_test.values
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
y_test = np_utils.to_categorical(encoded_y)

print('Running test set...')
predicted_L2 = nn_model.evaluate(X_test, y_test, batch_size=32)
print(predicted_L2)

#Prediction
prediction = nn_model.predict(X_test, verbose=1)
print(prediction)
 
#
#Saves the history of the run
import datetime
loglog = history.history
log_file = open('C:/Users/iseliner/Documents/programming/logfile.txt', 'a')
log_file.write(str(datetime.datetime.now()) + '\n')
log_file.write('Training loss: ' + str(loglog['loss']) + '\n')
log_file.write('Training acc: ' + str(loglog['acc']) + '\n')
log_file.write('Test set: ' + str(predicted_L2) + '\n \n')
log_file.close()


#RESULTS COUNTING. NOT FUNCTIONALITY FOR THE MODEL ___________________________
matrix_labels = ['ARA', 'CHI', 'FRE', 'GER', 'HIN', 'ITA', 'JPN', 'KOR',
                 'SPA', 'TEL', 'TUR']

predictionslist = []
x = 0
while x in range(len(prediction)):
    category_pred = np.argmax(prediction[x])
    predictionslist.append(matrix_labels[category_pred])
    x += 1
    
predictionslist_groundtruth = []
x = 0
while x in range(len(y_test)):
    category_pred = np.argmax(y_test[x])
    predictionslist_groundtruth.append(matrix_labels[category_pred])
    x += 1

#MAKES CONFUSION MATRIX|||||||||||||||||||||||||||||||||||||||||||||||||||||||
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(predictionslist, predictionslist_groundtruth)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=matrix_labels,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=matrix_labels, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

#Checks if GPU is usable
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

'''
#Fetching i-vectors from distributed json file
ivector = []
with open('C:/Users/iseliner/Documents/programming/' +
          '/data/data/ivectors/train/ivectors.json') as data_file:    
    data = json.load(data_file)
    for x in data:
        ivector.append(data[x])

ivector = np.array(ivector)
scaler = MinMaxScaler(copy=False,feature_range=(0, 1))
scaler.fit_transform(ivector)
'''

'''
scaler2 = MinMaxScaler(copy=False,feature_range=(0, 1))
new_df = pd.DataFrame(df)
#print('Building FastText...')
#
#Will train the scaling function
for i in range(new_df.size-1):
    vector = new_df.iloc[1,i]
    print(type(vector))
    for j in vector:
        scaler2.fit(j)

X_train = []
for vector in new_df:
    X_train.append(scaler2.transform(vector))
    '''
    