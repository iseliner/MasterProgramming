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
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel


#IMPORTS DATASET AND LABELS
print('Loading data...')

train_label = pd.read_csv('./data/data/labels/train/labels.train.csv')

dataset_path = ('./data/data/essays/train/original/')

speech_path = ('./data/data/speech_transcriptions/train/original/')

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
        
def makeseqvec(path, listname):
    for file in os.listdir(path):
        read_file = open(path + str(file))
        row = read_file.read()
        read_file.close()
        listname.append(row)

    
#Slices elements that are too long and appends the shorter version
def slicefiles(target_df):
    counter = 0
    for essay in target_df:
        if len(essay) > vector_len:
            old_new_sen = essay[0:vector_len]
            target_df[counter] = old_new_sen
        counter += 1

                
                
#11000 elements, each containing all words in their respective essay
#Make the label vectors: y_train(11000,11)
y = train_label.values
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
y_train = np_utils.to_categorical(encoded_y)

#ESSAY ````````````````````````````````````````````````````````````````````
vector_len = 350
essay_df = []
makeseq(dataset_path, essay_df)

#makelower(df)
slicefiles(essay_df)

#sg=1 is skip-gram, cbow otherwise
print('Building Tfidf...')
temp_df = []
for seq in essay_df:
    sen = ''
    for word in seq:
        sen += ' ' + str(word)
    temp_df.append(sen)

essay_transformer = TfidfVectorizer(lowercase=False, max_features=600)
essay_transformer = essay_transformer.fit(temp_df)

X_train_essay = essay_transformer.transform(temp_df).todense()

print('Setting up x_train and y_train...')
essay_clf = ExtraTreesClassifier()
essay_clf = essay_clf.fit(X_train_essay, y_train)
select_essay_model = SelectFromModel(essay_clf, prefit=True)
X_train_essay = select_essay_model.transform(X_train_essay)

X_train_essay = np.array(X_train_essay)
X_train_essay = np.reshape(X_train_essay, (X_train_essay.shape[0], X_train_essay.shape[1], 1))

#IVECTOR```````````````````````````````````````````````````````````````````````
print('Preparing data for input into the model...')
#Fetching i-vectors from distributed json file
ivector = []
with open('./data/data/ivectors/train/ivectors.json') as data_file:    
    data = json.load(data_file)
    for x in data:
        ivector.append(data[x])

ivector = np.array(ivector)

ivec_lda = LinearDiscriminantAnalysis()
ivec_lda.fit(ivector, encoded_y)

X_train_ivec = ivec_lda.transform(ivector)

#ivec_clf = ExtraTreesClassifier()
#ivec_clf = ivec_clf.fit(X_train_ivec, y_train)
#select_ivec_model = SelectFromModel(ivec_lda, prefit=True)
#X_train_ivec = select_ivec_model.transform(X_train_ivec)

#Creates the TRAINING INPUT for the model  
X_train_ivec = np.reshape(X_train_ivec, (X_train_ivec.shape[0], X_train_ivec.shape[1], 1))

#SPEECH ``````````````````````````````````````````````````````````````````````
#Creates the dataset (!)
vector_len = 150
speech_df = []
makeseq(speech_path, speech_df)

slicefiles(speech_df)

temp_df = []
for seq in speech_df:
    sen = ''
    for word in seq:
        sen += ' ' + str(word)
    temp_df.append(sen)

speech_transformer = TfidfVectorizer(lowercase=True, max_features=200)
speech_transformer = speech_transformer.fit(temp_df)

X_train_speech = speech_transformer.transform(temp_df).todense()

speech_clf = ExtraTreesClassifier()
speech_clf = speech_clf.fit(X_train_speech, y_train)
select_speech_model = SelectFromModel(speech_clf, prefit=True)
X_train_speech = select_speech_model.transform(X_train_speech)

X_train_speech = np.reshape(X_train_speech, (X_train_speech.shape[0], X_train_speech.shape[1], 1))
            

## MODEL ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
#from keras.layers import LSTM
from keras.layers import GRU
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

#Essay word input model
visible1 = Input(shape=(X_train_essay.shape[1], 1))
rnn11 = GRU(150, return_sequences=True)(visible1)
rnn12 = GRU(150, return_sequences=False)(rnn11)
dense1 = Dense(20, activation='relu')(rnn12)

#Speech transcript word input model
visible2 = Input(shape=(X_train_speech.shape[1], 1))
lstm21 = GRU(30, return_sequences=True)(visible2)
lstm22 = GRU(30, return_sequences=True)(lstm21)
lstm23 = GRU(30, return_sequences=False)(lstm22)
dense2 = Dense(10, activation='relu')(lstm23)

#i-vector input model
visible3 = Input(shape=(X_train_ivec.shape[1], 1))
lstm31 = GRU(10, return_sequences=True)(visible3)
lstm32 = GRU(10, return_sequences=False)(lstm31)
dense3 = Dense(10, activation='relu')(lstm32)

#Merge input-models
merge = concatenate([dense1, dense2, dense3])

output = Dense(11, activation='softmax')(rnn12)

model = Model(inputs=[visible1], outputs=output)

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'rmsprop', 
              metrics=['accuracy'])

filepath = './saved_models/GRUensemble_150epoch_lowercase_tfidf_essay_len350_featurecap600_2hidden150.h5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, 
                             save_best_only=True, mode='min')
earlystop = EarlyStopping(patience=1, monitor='loss')
callbacks_list = [checkpoint, earlystop]

print(model.summary())
#plot_model(model,to_file='LSTMensemble.png')

history = model.fit([X_train_essay], y_train, epochs=150, batch_size=60, 
                       callbacks=callbacks_list)


##TEST
#////////////////////////////////////////////////////////////////////////
#TESTING ________________________________________________________________
test_label = pd.read_csv('./data/data/labels/dev/labels.dev.csv')

essay_test_path = ('./data/data/essays/dev/original/')

speech_test_path = ('./data/data/speech_transcriptions/dev/original/')

#           changes labels index to the test_taker_id
test_label = test_label.set_index('test_taker_id')
#           drop the prompts, since we don't use them
test_label = test_label.drop('speech_prompt', axis=1)
test_label = test_label.drop('essay_prompt', axis=1)
        
print('Setting final labels') #|||||||||||||||||||||||||||||
y = test_label.values
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
y_test = np_utils.to_categorical(encoded_y)

#11000 elements, each containing all words in the essay
print('Initializing essay data for testing...') #|||||||||||||||||||||||||||||
vector_len = 350
essay_test_df = []
makeseq(essay_test_path, essay_test_df)
slicefiles(essay_test_df)

temp_df = []
for seq in essay_test_df:
    sen = ''
    for word in seq:
        sen += word
    temp_df.append(sen)

X_test_essay = essay_transformer.transform(temp_df).todense()

X_test_essay = select_essay_model.transform(X_test_essay)

X_test_essay = np.array(X_test_essay)
X_test_essay = np.reshape(X_test_essay, (X_test_essay.shape[0], X_test_essay.shape[1], 1))

print('Initializing ivector data for testing...') #|||||||||||||||||||||||||||||
#Fetching i-vectors from distributed json file
test_ivector = []
with open('./data/data/ivectors/dev/ivectors.json') as data_file:    
    data = json.load(data_file)
    for x in data:
        test_ivector.append(data[x])

X_test_ivector = np.array(test_ivector)
X_test_ivector = ivec_lda.transform(X_test_ivector)
X_test_ivector = np.reshape(X_test_ivector, (X_test_ivector.shape[0], X_test_ivector.shape[1], 1))

#SPEECH
print('Initializing speech data for testing...') #|||||||||||||||||||||||||||||
vector_len = 150
speech_test_df = []
makeseq(speech_test_path, speech_test_df)
slicefiles(speech_test_df)

speech_test_df = []
for seq in speech_test_df:
    sen = ''
    for word in seq:
        sen += word
    temp_df.append(sen)

X_test_speech = speech_transformer.transform(temp_df).todense()
X_test_speech = select_speech_model.transform(X_test_speech)


X_test_speech = np.reshape(X_test_speech, (X_test_speech.shape[0], X_test_speech.shape[1], 1))


print('Running test set...') #|||||||||||||||||||||||||||||
predicted_L2 = model.evaluate([X_test_essay], y_test, batch_size=32)
print(predicted_L2)

#Prediction
prediction = model.predict([X_test_essay], verbose=1)
print(prediction)


