# -*- coding: utf-8 -*-

#Import the libraries
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import json
#import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
#from sklearn.feature_selection import SelectFromModel
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.feature_extraction.text import CountVectorizer

from keras.utils import np_utils
#from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
#from keras.preprocessing import sequence

#import gensim
#import nltk
#from nltk.corpus import brown as brown
#from gensim.models import Word2Vec
from gensim.models import KeyedVectors
#from gensim.models import FastText

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#nltk.download('brown')

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
fourgram_essay_path = ('C:/Users/iseliner/Documents/programming/' +
                       'features/ngram/essays/4gram/')

#           changes labels index to the test_taker_id
train_label = train_label.set_index('test_taker_id')
#           drop the prompts, since we don't use them
train_label = train_label.drop('speech_prompt', axis=1)
train_label = train_label.drop('essay_prompt', axis=1)

vector_len = 400

#Function which imports a datasets from a path and puts it into a list
def makeseq(path, listname):
    for file in os.listdir(path):
        read_file = open(path + str(file))
        row = read_file.read().split()
        read_file.close()
        listname.append(row)

#Makes vectors from the lists        
def makevectors(listname, vectorlist, target_model):
    for essay in listname:
        essay_sen = [np.zeros(100)] * vector_len
        for i,word in enumerate(essay):
            if word in target_model.wv.vocab:
                enc = target_model.wv[word]
                essay_sen[i] = enc
            else:
                target_model.build_vocab([word], update=True)
    vectorlist.append(essay_sen)

#Appends new dataset features, slicing them to fit to the already existing one
def appenddata(target_data, target_df, label_list):
    makeseq(target_data, target_df)
    x = 0
    while x < 11000:
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
        counter += 1
        
'''    
def vectorizedata(target_df, target_model):
    X = []
    for seq in target_df:
        sequence = [np.zeros(100)] * vector_len
        for i,word in enumerate(seq):
            if word in model.wv.vocab:
                enc = model.wv[word]
                sequence[i] = enc
            else:
                enc = model.wv.most_similar(word)
                sequence[i] = enc
        X.append(sequence)
    return X
'''

def vectorizedata2(target_df, target_model):
    X = []
    for seq in target_df:
        sequence = []
        for word in seq:
            if word in target_model.wv.vocab:
                enc = target_model.wv[word]
                sequence.append(enc)
            else:
                enc = target_model.wv.most_similar(word)
                sequence.append(enc)
        X.append(sequence)
    return X

def padseq(target_df):
    temp_array = np.array([0]*100)
    i = 0
    for seq in target_df:
        if len(seq) < vector_len:
            new_seq = seq
            while len(seq) < vector_len:
                new_seq = np.concatenate(new_seq,temp_array)
            target_df[i] = new_seq
        i += 1
            

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
appenddata(bigram_essay_path,df,label_list)
appenddata(trigram_essay_path,df,label_list)
appenddata(fourgram_essay_path,df,label_list)

#Slice elements in the dataset that are longer than the desired length
slicefiles(df)

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

#sg=1 is skip-gram, cbow otherwise
print('Building Word2Vec...')
#model = Word2Vec(sentences=df, size=100, min_count=0, workers=6, window=5,sg=0)
#word_vectors = model.wv
#word_vectors.save('C:/Users/iseliner/Documents/programming/word2vectors.bin')

#Use this when the word vector is final, and comment out the model building above
model_load = KeyedVectors.load('C:/Users/iseliner/' +
                    'Documents/programming/word2vectors.bin')

#print('Building FastText...')
#model = FastText(df, size=100, min_count=0, workers=5, window=5,sg=0)


print('Setting up x_train and y_train...')
#df = vectorizedata(df)
df = vectorizedata2(df, model_load)

X_train = padseq(df)
scaler2 = MinMaxScaler(copy=False,feature_range=(0, 1))
new_df = pd.DataFrame(df)
for i in range(new_df.size-1):
    vector = new_df.iloc[i]
    print(vector)
    scaler2.fit(vector)
for vector in new_df:
    X_train.append(scaler2.transform(vector))
padseq(X_train) 

X_train = np.array(df)
X_train = np.reshape(X_train, (X_train.shape))

print(new_df[0])
print(X_train[1])

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

#MODEL _________________________________________________________________
print('Creating model...')

#Initalizing the RNN
nn_model = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
nn_model.add(LSTM(200, return_sequences = True, 
               input_shape = (X_train.shape[1], X_train.shape[2])))
nn_model.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
nn_model.add(LSTM(200, return_sequences = False))
nn_model.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
#nn_model.add(LSTM(150, return_sequences = False))
#nn_model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
#nn_model.add(LSTM(150))

# Adding the output layer
nn_model.add(Dense(11, activation='softmax'))

#Should try the RMSPROP as optimizer
# Compiling the RNN
nn_model.compile(loss = 'categorical_crossentropy', 
              optimizer = 'rmsprop', 
              metrics=['accuracy'])


# Fitting the RNN to the Training set
print('Fitting data to the model...')
history = nn_model.fit(X_train, y_train, epochs = 20, batch_size = 95)
print('Training complete!')

#For later scoring
#score = model.evaluate(x_test, y_test, batch_size=16)

#currentLearningRate = K.get_value(nn_model.optimizer.lr)

#////////////////////////////////////////////////////////////////////////
#TESTING ________________________________________________________________
test_label = pd.read_csv('./data/data/labels/dev/labels.dev.csv')
test_path = ('./data/data/essays/dev/original/')

#           changes labels index to the test_taker_id
test_label = test_label.set_index('test_taker_id')
#           drop the prompts, since we don't use them
test_label = test_label.drop('speech_prompt', axis=1)
test_label = test_label.drop('essay_prompt', axis=1)

print('Initializing test set labels')
y = test_label.values
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
y_test = np_utils.to_categorical(encoded_y)

#11000 elements, each containing all words in the essay
test_df = []
makeseq(test_path, test_df)

print('Initializing test data')
test_df = vectorizedata2(test_df, model_load)

x = np.array(test_df)
X_test = np.reshape(x, (x.shape))

print('Running test set...')
predicted_L2 = nn_model.evaluate(X_test, y_test, batch_size=32)
print(predicted_L2)

#Prediction
prediction = nn_model.predict(X_test, verbose=1)
print(prediction)



'''
#
#Saves the history of the run
import datetime
loglog = history.history
log_file = open('logfile.txt', 'a')
log_file.write(str(datetime.datetime.now()) + '\n')
log_file.write('Training loss: ' + str(history.history['loss']) + '\n')
log_file.write('Training acc: ' + str(history.history['acc']) + '\n')
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

#MAKES CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

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
'''
#Checks if GPU is usable
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())