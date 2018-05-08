##PREPROCESSING and LOADING OF DATA
import os
import gc
import json
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

speech_path = ('C:/Users/iseliner/Documents/programming' +
                    '/data/data/speech_transcriptions/train/original/')
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
            #if len(essay) > vector_len*2:
            #    new_essay = essay[vector_len:vector_len*2]
            #else:
            #    new_essay = essay[vector_len:len(essay)]
            old_new_sen = essay[0:vector_len]
            #label = label_list[counter]
            target_df[counter] = old_new_sen
            #target_df.append(new_essay)
            #label_list.append(label)
            #if len(essay) > vector_len*3:
            #    new_essay = essay[vector_len*2:vector_len*3]
            #    target_df.append(new_essay)
            #    label_list.append(label)
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

#makelower(df)
slicefiles(df)


#sg=1 is skip-gram, cbow otherwise
print('Building Word2Vec...')

w2v = Word2Vec(sentences=df, size=embed_len, min_count=2, workers=6, window=5,sg=0)

print('Preparing data for input into the model...')
#Creates the TRAINING INPUT for the model  

#X_train = np.reshape(x, (x.shape))

label_train = pd.DataFrame(label_list)
#Make the label vectors: y_train(11000,11)
y = label_train.values
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
y_train_essay = np_utils.to_categorical(encoded_y)

print('Setting up x_train and y_train...')

df = vectorizedata(df, w2v)
X_train = padvectors(df)
#X_train = np.array(X_train)

del df
gc.collect()

#select_model = ExtraTreesClassifier()
#select_model.fit(X_train, y_train)
#X_train = select_model.transform(X_train)

X_train_essay = np.array(X_train)
X_train_essay = np.reshape(X_train_essay, (X_train_essay.shape))

X_train_speech = X_train_essay
y_train_speech = y_train_essay

#IVECTOR
print('Preparing data for input into the model...')
#Fetching i-vectors from distributed json file
ivector = []
with open('C:/Users/iseliner/Documents/programming/' +
          '/data/data/ivectors/train/ivectors.json') as data_file:    
    data = json.load(data_file)
    for x in data:
        ivector.append(data[x])
     
y = train_label.values
encoder = LabelEncoder()
encoder.fit(y)
ivec_encoded_y = encoder.transform(y)
y_train_ivec = np_utils.to_categorical(ivec_encoded_y)

ivector = np.array(ivector)
clf = LinearDiscriminantAnalysis()
clf.fit(ivector, ivec_encoded_y)
X_new = clf.transform(ivector)

#Creates the TRAINING INPUT for the model  
X_train_ivec = np.reshape(X_new, (X_new.shape[0], X_new.shape[1], 1))

#SPEECH
#Creates the dataset (!)
vector_len = 100
speech_df = []
makeseq(speech_path, speech_df)

#makelower(df)
slicefiles(speech_df)
print('Building Word2Vec...')

w2v_speech = Word2Vec(sentences=speech_df, size=embed_len, min_count=2, workers=6, window=5,sg=0)

speech_df = vectorizedata(speech_df, w2v_speech)
X_train_speech = padvectors(speech_df)
#X_train = np.array(X_train)

del speech_df
gc.collect()

#select_model = ExtraTreesClassifier()
#select_model.fit(X_train, y_train)
#X_train = select_model.transform(X_train)

X_train_speech = np.array(X_train_speech)
X_train_speech = np.reshape(X_train_speech, (X_train_speech.shape))
            

## MODEL ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint

#Essay word input model
visible1 = Input(shape=(X_train_essay.shape[1], X_train_essay.shape[2]))
lstm11 = LSTM(40, return_sequences=True)(visible1)
lstm12 = LSTM(40, return_sequences=False)(lstm11)
#lstm12 = LSTM(68)(lstm11)
dense1 = Dense(40, activation='relu')(lstm12)

#Speech transcript word input model
visible2 = Input(shape=(X_train_speech.shape[1], X_train_speech.shape[2]))
lstm21 = LSTM(10, return_sequences=False)(visible2)
#lstm22 = LSTM(10)(lstm21)
dense2 = Dense(10, activation='relu')(lstm21)

#i-vector input model
visible3 = Input(shape=(X_train_ivec.shape[1], 1))
lstm31 = LSTM(10, return_sequences=True)(visible3)
lstm32 = LSTM(10, return_sequences=False)(lstm31)
#lstm32 = LSTM(10)(lstm31)
dense3 = Dense(10, activation='relu')(lstm32)

#Merge input-models
merge = concatenate([dense1, dense2, dense3])

#interpretation
hidden1 = Dense(20, activation='relu')(merge)
output = Dense(11, activation='softmax')(hidden1)

model = Model(inputs=[visible1, visible2, visible3], outputs=output)

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'rmsprop', 
              metrics=['accuracy'])

filepath = 'C:/Users/iseliner/Documents/programming/saved_models/LSTMensemble_10epoch.h5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, 
                             save_best_only=True, mode='min')
callbacks_list = [checkpoint]

print(model.summary())
#plot_model(model,to_file='LSTMensemble.png')

history = model.fit([X_train_essay, X_train_speech, X_train_ivec], y_train_essay, epochs=20, batch_size=60, 
                       callbacks=callbacks_list)


##TEST
#////////////////////////////////////////////////////////////////////////
#TESTING ________________________________________________________________
test_label = pd.read_csv('C:/Users/iseliner/Documents/programming/data/data/labels/dev/labels.dev.csv')
essay_test_path = ('C:/Users/iseliner/Documents/programming/data/data/essays/dev/original/')
speech_test_path = ('C:/Users/iseliner/Documents/programming/data/data/speech_transcriptions/dev/original/')

speech_test_path = ('C:/Users/iseliner/Documents/programming' +
                    '/data/data/speech_transcriptions/dev/original/')

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
vector_len = 200
essay_test_df = []
makeseq(essay_test_path, essay_test_df)
slicetestfiles(essay_test_df)
essay_test_df = vectorizedata(essay_test_df, w2v)
essay_test_df = padvectors(essay_test_df)

X_test_essay = np.array(essay_test_df)
X_test_essay = np.reshape(X_test_essay, (X_test_essay.shape))

print('Setting final labels')
label_test = pd.DataFrame(test_label_list)
y = label_test.values
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
y_test = np_utils.to_categorical(encoded_y)

print('Preparing data for testing...')
#Fetching i-vectors from distributed json file
test_ivector = []
with open('C:/Users/iseliner/Documents/programming/' +
          '/data/data/ivectors/dev/ivectors.json') as data_file:    
    data = json.load(data_file)
    for x in data:
        test_ivector.append(data[x])

test_ivector = np.array(test_ivector)
test_new = clf.transform(test_ivector)
X_test_ivec = np.reshape(test_new, (test_new.shape[0], test_new.shape[1], 1))

#SPEECH
vector_len = 100
speech_test_df = []
makeseq(speech_test_path, speech_test_df)
slicetestfiles(speech_test_df)
speech_test_df = vectorizedata(speech_test_df, w2v_speech)
speech_test_df = padvectors(speech_test_df)

X_test_speech = np.array(speech_test_df)
X_test_speech = np.reshape(X_test_speech, (X_test_speech.shape))

y_test_speech = y_test
y_test_ivec = y_test

print('Running test set...')
predicted_L2 = model.evaluate([X_test_essay, X_test_speech, X_test_ivec], [y_test], batch_size=32)
print(predicted_L2)

#Prediction
prediction = model.predict([X_test_essay, X_test_speech, X_test_ivec], verbose=1)
print(prediction)

#from sklearn.metrics import f1_score
#f1 = f1_score(y_test, prediction)
#print(f1)
 
#
#Saves the history of the run
import matplotlib.pyplot as plt
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


