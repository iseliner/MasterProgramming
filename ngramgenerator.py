# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 13:19:24 2018

@author: iseliner
"""
import os

dataset_path = ('C:/Users/iseliner/Documents/programming/' +
                'data/data/essays/train/original/')

target_path = ('C:/Users/iseliner/Documents/programming/' +
                'features/ngram/essays/6gram/')

target_char_path = ('C:/Users/iseliner/Documents/programming/' +
                    'features/char_ngram/essays/8gram/')

def makeseq(path, list_name):
    for file in os.listdir(path):
        read_file = open(path + str(file))
        row = read_file.read().split()
        read_file.close()
        list_name.append(row)
        
def makeseq2(path, list_name):
    for file in os.listdir(path):
        read_file = open(path + str(file))
        row = read_file.read()
        read_file.close()
        list_name.append(row)
        
def writefile(file_name, list_name):
    for item in list_name:
        file_name.write(str(item))

df = []
df_whole = []
makeseq(dataset_path, df)
makeseq2(dataset_path, df_whole)

def word_ngrams(input_list, n):
    counter = 1
    for essay in input_list:
        gram_list = []
        gram_list.append(([essay[i:i+n] for i in range(len(essay)-n+1)]))
        
        doc = open(target_path + str(counter) + '.txt', 'w')
        writefile(doc, gram_list)
        doc.close()
        counter += 1
    return gram_list

word_gram = word_ngrams(df, 6)

def char_ngrams(input_list, n):
    counter = 1
    for essay in input_list:
        gram_list = []
        gram_list.append([essay[i:i+n] for i in range(len(essay)-n+1)])
        
        doc = open(target_char_path + str(counter) + '.txt', 'w')
        writefile(doc, gram_list)
        doc.close()
        counter += 1
    return gram_list
        
char_gram = char_ngrams(df_whole,8)



