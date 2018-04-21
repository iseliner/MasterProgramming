

#Import the libraries
import tensorflow as tf
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
#from sklearn.feature_extraction.text import TfidfVectorizer
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
from gensim.models import Word2Vec

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#nltk.download('brown')

#IMPORTS DATASET AND LABELS
print('Loading data...')

train_label = pd.read_csv('C:/Users/iseliner/Documents/programming/' +
                                  'data/data/labels/train/labels.train.csv')

dataset_path = ('C:/Users/iseliner/Documents/programming/' +
                'data/data/essays/train/original/')

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

#Function which imports a datasets from a path and puts it into a list
def makeseq(path, listname):
    for file in os.listdir(path):
        read_file = open(path + str(file))
        row = read_file.read().split()
        read_file.close()
        listname.append(row)

#Makes vectors from te lists        
def makevectors(listname, vectorlist):
    for essay in listname:
        essay_sen = [np.zeros(100)] * 350
        for i,word in enumerate(essay):
            if word in model.wv.vocab:
                enc = model.wv[word]
                essay_sen[i] = enc
            else:
                model.build_vocab([word], update=True)
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
        if len(essay) > 350:
            if len(essay) > 700:
                new_essay = essay[350:700]
            else:
                new_essay = essay[350:len(essay)]
            old_new_sen = essay[0:350]
            label = label_list[counter]
            target_df[counter] = old_new_sen
            target_df.append(new_essay)
            label_list.append(label)
        counter += 1
    
#11000 elements, each containing all words in their respective essay
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

#Trains the world2vec model to vectorize data    
'''
print('Initiate Word2Vec sentences')
brown_sents = []
for sents in brown.tagged_sents():
    brown_sents.append(sents)
for sents in brown.sents():
    brown_sents.append(sents)
x = 0
while x in range(len(brown_sents)):
    z = 0
    for y in brown_sents[x]:
        temp = str(y)
        brown_sents[x][z] = temp
        z += 1
    x += 1    
'''

print('Building Word2Vec...')
model = Word2Vec(size=100, min_count=0, workers=3)
model.build_vocab(df)
#total_examples = model.corpus_count
print('Training Word2Vec')
model.train(df, total_examples=11000, epochs=30)


print('Setting up x_train and y_train...')
X = []
for essay in df:
    essay_sen = [np.zeros(100)] * 350
    for i,word in enumerate(essay):
        if word in model.wv.vocab:
            enc = model.wv[word]
            essay_sen[i] = enc
    X.append(essay_sen)
        

#Creates the TRAINING INPUT for the model  
x = np.array(X)

X_train = np.reshape(x, (x.shape))

label_train = pd.DataFrame(label_list)
#Make the label vectors: y_train(11000,11)
y = label_train.values
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
y_train = np_utils.to_categorical(encoded_y)
y_train = pd.DataFrame()

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

#////////////////////////////////////////////////////////////////////////
#TESTING ________________________________________________________________
test_label = pd.read_csv('C:/Users/iseliner/Documents/programming/' +
                                  'data/data/labels/dev/labels.dev.csv')
test_path = ('C:/Users/iseliner/Documents/programming/' +
                'data/data/essays/dev/original/')

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
test = []
counter = 0
for essay in test_df:
    if len(essay) > 350:
        old_new_sen = essay[0:350]
        essay = old_new_sen
    essay_sen = [np.zeros(100)] * 350
    for i,word in enumerate(essay):
        if word in model.wv.vocab:
          enc = model.wv[word]
          essay_sen[i] = enc
#        else:
#            model.build_vocab([word], update=True)
    test.append(essay_sen)

x = np.array(test)
X_test = np.reshape(x, (x.shape))

print('Running test set...')
predicted_L2 = nn_model.evaluate(X_test, y_test, batch_size=32)
print(predicted_L2)

#Prediction
prediction = nn_model.predict(X_test, verbose=1)
print(prediction)

#Saves the history of the run
import datetime
loglog = history.history
log_file = open('C:/Users/iseliner/Documents/programming/logfile.txt', 'a')
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

#Checks if GPU is usable
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())







































