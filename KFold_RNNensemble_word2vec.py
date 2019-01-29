'''
Created by Iselin Bjørnsgaard Erikssen for her master thesis in databases and search
 at NTNU spring 2018.

The current state of this python file is the same state as the one used for the
top performing model found through the experiments, which is described within 
the master thesis. Some unused functions have been kept intact for the purpose 
of further inspections or explanations to the results underway.
'''

##PREPROCESSING and LOADING OF DATA
import os
import json
import numpy as np
import pandas as pd
from nltk import ngrams
from gensim.models import Word2Vec
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#IMPORTS DATASET AND LABELS
print('Loading data...')

train_label = pd.read_csv('C:/Users/iseliner/Documents/programming/' +
                          '/data/data/labels/train/labels.train.csv')

essay_path = ('C:/Users/iseliner/Documents/programming/' +
                '/data/data/essays/train/original/')

speech_path = ('C:/Users/iseliner/Documents/programming' +
                    '/data/data/speech_transcriptions/train/original/')

#           changes labels index to the test_taker_id
train_label = train_label.set_index('test_taker_id')
#           drop the prompts, since we don't use them
train_label = train_label.drop('speech_prompt', axis=1)
train_label = train_label.drop('essay_prompt', axis=1)

#Function which imports a datasets from a path and puts it into a list
def makeseq(path, listname):
    for file in os.listdir(path):
        read_file = open(path + str(file))
        row = read_file.read().split()
        read_file.close()
        listname.append(row)
    
#Slices elements that are too long to the specified vector length
def slicefiles(target_df, vector_len):
    counter = 0
    for essay in target_df:
        if len(essay) > vector_len:
            old_new_sen = essay[0:vector_len]
            target_df[counter] = old_new_sen
        counter += 1
        
#Pads ZEROS to the end of sequences shorter than specified vector length  
def zeropadvectors(target_df, vector_len, embed_len):
    X = []
    for seq in target_df:
        sequence = [np.zeros(embed_len)] * vector_len
        for i,word in enumerate(seq):                               
            sequence[i] = word
        X.append(sequence)
    return X

#Pads WORDS to the end of sequences shorter than specified vector length
def wordpadvectors(target_df, vector_len):
    X = []
    for seq in target_df:
        new_seq = seq
        if len(seq) < 2:
            X.append(new_seq)
        elif len(seq) < vector_len:
            for count in range(vector_len-len(seq)):
                new_seq.append(new_seq[count])
            X.append(new_seq)
        else:
            X.append(new_seq)
    return X

#Applies the target word2vec model to transform all sequences in a list
def vectorizedata(target_df, target_model, embed_len):
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

#Converts all elements in the list to lowercase
def makelower(target_list):
    for x in range(len(target_list)):
        for y in range(len(target_list[x])):
            if target_list[x][y].isalpha():
                target_list[x][y] = target_list[x][y].lower()

#This function creates word n-grams of the chosen sequence list                
def wordngram(target_df, n):
    temp = []
    for essay in target_df:
        temp_seq = []
        temp_sen = ngrams(essay, n)
        for x in temp_sen:
            temp_seq.append(str(x))
        temp.append(temp_seq)
    return temp
        
        
#LABELS |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#Make the label vectors: y_train(11000,11)
y = train_label.values
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
y_train = np_utils.to_categorical(encoded_y)                
                
#ESSAY |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
essay_vector_len = 350
essay_embed_len = 150

essay_df = []
makeseq(essay_path, essay_df)
makelower(essay_df)
#essay_df = wordngram(essay_df, 3)

#sg=1 is skip-gram, cbow otherwise
print('Building Word2Vec for essays...')
w2v = Word2Vec(sentences=essay_df, size=essay_embed_len, min_count=5, workers=6, window=5,sg=0)

'''
If wordpadvectors is uncommented, the sequences will be naively padded with
words starting from the beginning of the sequence. 
'''
print('Setting up x train essay...')
slicefiles(essay_df, essay_vector_len)
#essay_df = wordpadvectors(essay_df, essay_vector_len)
essay_df = vectorizedata(essay_df, w2v, essay_embed_len)
essay_df = zeropadvectors(essay_df, essay_vector_len, essay_embed_len)
X_train_essay = np.array(essay_df)

X_train_essay = np.reshape(X_train_essay, (X_train_essay.shape))

 
#IVECTOR |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
print('Setting up x train ivector...')
#Fetching i-vectors from distributed json file
ivector = []
with open('C:/Users/iseliner/Documents/programming/' +
          '/data/data/ivectors/train/ivectors.json') as data_file:    
    data = json.load(data_file)
    for x in data:
        ivector.append(data[x])

'''
##This is the LDA transformer, which was intended to be a good method but was
##dropped in the end. Is still intact here for later inspection purposes
ivector = np.array(ivector)
clf = LinearDiscriminantAnalysis()
clf.fit(ivector, encoded_y)
X_train_ivec = clf.transform(ivector)
''' 
   
X_train_ivec = np.array(ivector)
X_train_ivec = np.reshape(X_train_ivec, (X_train_ivec.shape[0], X_train_ivec.shape[1], 1))


#SPEECH |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
speech_vector_len = 150
speech_embed_len = 100

speech_df = []
makeseq(speech_path, speech_df)
makelower(speech_df)

#sg=1 is skip-gram, cbow otherwise
print('Building Word2Vec for speech...')
w2v_speech = Word2Vec(sentences=speech_df, size=speech_embed_len, min_count=2, workers=6, window=5,sg=0)

slicefiles(speech_df, speech_vector_len)
#speech_df = wordpadvectors(speech_df, speech_vector_len)
speech_df = vectorizedata(speech_df, w2v_speech, speech_embed_len)
speech_df = zeropadvectors(speech_df, speech_vector_len, speech_embed_len)
X_train_speech = np.array(speech_df)

X_train_speech = np.array(X_train_speech)
X_train_speech = np.reshape(X_train_speech, (X_train_speech.shape))

## MODEL ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
print('Building model...')
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
#from keras.layers import LSTM
from keras.layers import GRU
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold

#Essay word input model
visible1 = Input(shape=(X_train_essay.shape[1], X_train_essay.shape[2]))
rnn11 = GRU(100, return_sequences=True, dropout=0.2)(visible1)
rnn12 = GRU(100, return_sequences=True, dropout=0.2)(rnn11)
rnn13 = GRU(100, return_sequences=False, dropout=0.2)(rnn12)
#rnn14 = GRU(200, dropout=0.2)(rnn13)
#rnn14 = GRU(100)(rnn13)
dense1 = Dense(20, activation='relu')(rnn13)


#Speech transcript word input model
visible2 = Input(shape=(X_train_speech.shape[1], X_train_speech.shape[2]))
rnn21 = GRU(60, return_sequences=True, dropout=0.2)(visible2)
rnn22 = GRU(60, dropout=0.2)(rnn21)
dense2 = Dense(20, activation='relu')(rnn22)

#i-vector input model
visible3 = Input(shape=(X_train_ivec.shape[1], 1))
rnn31 = GRU(60, return_sequences=False, dropout=0.2)(visible3)
dense3 = Dense(20, activation='relu')(rnn31)

#Merge input-models
merge = concatenate([dense1,dense2,dense3])

#interpretation
#hidden1 =Dense(30)(lstm13)
output = Dense(11, activation='softmax')(merge)

model = Model(inputs=[visible1, visible2, visible3], outputs=output)

filepath = 'C:/Users/iseliner/Documents/programming/saved_models/KFold5_60epoch_GRUensemble_essay350_essay3hidden100_150epoch_min5_cbow200_patience2_speech150_min2_cbow100_2hidden60_lowerspeech_loweressay_ivec1hidden60.h5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, 
                             save_best_only=True, mode='min')
early_stop = EarlyStopping(patience=2, monitor='loss', min_delta=0.01)
callbacks_list = [checkpoint, early_stop]

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'rmsprop', 
              metrics=['accuracy'])

print(model.summary())

y_train_alpha = y_train

kf = KFold(n_splits=5, shuffle=True)
split_indices=kf.split(X_train_essay)
for train, test in split_indices:
    train_labels = y_train_alpha[train]
    y_train = train_labels[:int(-0.1*len(train))]
    y_val = train_labels[int(-0.1*len(train)):]
    y_test = y_train_alpha[test]
    
    essay_train_data = X_train_essay[train]
    x_train_essay = essay_train_data[:int(-0.1*len(train))]
    x_val_essay = essay_train_data[int(-0.1*len(train)):]
    x_test_essay = X_train_essay[test]
    
    speech_train_data = X_train_speech[train]
    x_train_speech = speech_train_data[:int(-0.1*len(train))]
    x_val_speech = speech_train_data[int(-0.1*len(train)):]
    x_test_speech = X_train_speech[test]
    
    ivec_train_data = X_train_ivec[train]
    x_train_ivec = ivec_train_data[:int(-0.1*len(train))]
    x_val_ivec = ivec_train_data[int(-0.1*len(train)):]
    x_test_ivec = X_train_ivec[test]
    
    history = model.fit([x_train_essay, x_train_speech, x_train_ivec], y_train, validation_data=([x_val_essay, x_val_speech, x_val_ivec], y_val), epochs=60, batch_size=80, callbacks=callbacks_list)
    y_pred= model.predict([x_test_essay,], verbose=2) 

score = model.evaluate([X_train_essay, X_train_speech, X_train_ivec], y_train, verbose=1)


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

print('Setting labels')
y = test_label.values
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
y_test = np_utils.to_categorical(encoded_y)

#11000 elements, each containing all words in the essay
print('Initializing essay test data')
essay_test_df = []
makeseq(essay_test_path, essay_test_df)
slicefiles(essay_test_df, essay_vector_len)
makelower(essay_test_df)
#essay_test_df = wordngram(essay_test_df, 3)

#essay_test_df = wordpadvectors(essay_test_df, essay_vector_len)
essay_test_df = vectorizedata(essay_test_df, w2v, essay_embed_len)
essay_test_df = zeropadvectors(essay_test_df, essay_vector_len, essay_embed_len)

X_test_essay = np.array(essay_test_df)
X_test_essay = np.reshape(X_test_essay, (X_test_essay.shape))

#IVEC
print('Preparing data for testing...')
#Fetching i-vectors from distributed json file
test_ivector = []
with open('C:/Users/iseliner/Documents/programming/' +
          '/data/data/ivectors/dev/ivectors.json') as data_file:    
    data = json.load(data_file)
    for x in data:
        test_ivector.append(data[x])

test_new = np.array(test_ivector)
#test_new = clf.transform(test_new)
X_test_ivec = np.reshape(test_new, (test_new.shape[0], test_new.shape[1], 1))

print('Initializing speech test data')
#SPEECH
speech_test_df = []
makeseq(speech_test_path, speech_test_df)
slicefiles(speech_test_df, speech_vector_len)
makelower(speech_test_df)

#speech_test_df = wordpadvectors(speech_test_df, speech_vector_len)
speech_test_df = vectorizedata(speech_test_df, w2v_speech, speech_embed_len)
speech_test_df = zeropadvectors(speech_test_df, speech_vector_len, speech_embed_len)

X_test_speech = np.array(speech_test_df)
X_test_speech = np.reshape(X_test_speech, (X_test_speech.shape))

print('Running test set...')
predicted_L2 = model.evaluate([X_test_essay, X_test_speech, X_test_ivec], [y_test], batch_size=32)
print(predicted_L2)

#Prediction
prediction = model.predict([X_test_essay, X_test_speech, X_test_ivec], verbose=1)
print(prediction)


#
#
#
#Saves the history of the run
import matplotlib.pyplot as plt
import datetime
loglog = history.history
log_file = open('C:/Users/iseliner/Documents/programming/logfile.txt', 'a')
log_file.write(str(datetime.datetime.now()) + '\n')
log_file.write(str(filepath) + '\n')
log_file.write('Training loss: ' + str(loglog['loss']) + '\n')
log_file.write('Training acc: ' + str(loglog['acc']) + '\n')
log_file.write('Test set: ' + str(predicted_L2) + '\n')


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
    
from sklearn.metrics import f1_score
f1 = f1_score(predictionslist_groundtruth, predictionslist, average='macro')
print('F1:' + str(f1))
log_file.write('F1: ' + str(f1) + '\n \n')
log_file.close()

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


