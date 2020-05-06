import numpy as np
import pandas as pd
import re
import sys

from nltk.stem import PorterStemmer 
# from nltk.tokenize import word_tokenize 
   
ps = PorterStemmer() 

# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('punkt')
# stop = set(stopwords.words('english'))
# print(stop)

# (['she', 'be', 'above', 'your', 'are', 'just', 'what', 'doesn', 'you', 'while', 'yourself',
# "weren't", 'having', 'only', 'in', "doesn't", 'don', "won't", "you've", 'them', 'her', 't', "you'll",
# 're', 'up', 'been', 'how', 'very', 'there', "don't", 'needn', 'hasn', 'wouldn', 'about', 'has',
# 'myself', 'further', "haven't", 'which', 'doing', 'haven', 'his', 've', 'as', 'same', 'couldn', 's',
# 'between', 'theirs', 'then', 'such', "aren't", 'being', 'isn', 'if', 'those', 'they', 'were', 'of',
# 'can', 'why', 'wasn', 'through', 'him', 'at', 'ain', 'i', 'against', 'was', 'where', "needn't",
# 'that', 'did', 'do', 'until', 'their', 'o', 'when', 'over', 'own', "hadn't", 'yours', 'aren', 'into',
# 'so', 'nor', 'had', 'didn', "isn't", 'an', "you'd", 'he', 'mustn', 'shan', 'themselves', 'me',
# "should've", 'out', "wasn't", 'and', 'no', 'other', 'because', "shan't", 'will', 'it', 'again',
# 'yourselves', 'or', 'should', 'y', "couldn't", 'both', 'during', 'on', 'all', 'who', 'below',
# "mightn't", "she's", 'our', 'once', 'ours', 'does', 'the', 'weren', 'we', 'from', 'shouldn', 'its',
# 'mightn', 'whom', 'itself', 'after', "wouldn't", 'few', 'ma', 'll', 'but', "didn't", 'by', 'any',
# 'hers', 'to', 'now', "mustn't", 'have', 'each', 'down', 'this', 'some', "that'll", 'himself', 'hadn',
# 'd', 'a', 'than', 'am', "shouldn't", 'with', 'my', 'most', 'm', 'herself', 'under', 'these', "hasn't",
# 'for', 'off', 'more', "it's", 'too', 'won', "you're", 'before', 'ourselves', 'not', 'here', 'is'])

stop = {'she', 'be', 'above', 'your', 'are', 'just', 'what', 'doesn', 'you', 'while', 'yourself',
"weren't", 'having', 'only', 'in', "doesn't",                "you've", 'them', 'her', 't', "you'll",
're', 'up', 'been', 'how', 'very', 'there',            'needn', 'hasn',            'about', 'has',
'myself', 'further',           'which', 'doing', 'haven', 'his', 've', 'as', 'same', 'couldn', 's',
'between', 'theirs', 'then', 'such', "aren't", 'being', 'isn', 'if', 'those', 'they', 'were', 'of',
'can', 'why',       'through', 'him', 'at',       'i',              'was', 'where',
'that', 'did', 'do', 'until', 'their', 'o', 'when', 'over', 'own',          'yours', 'aren', 'into',
'so',           'had', 'didn',             'an', "you'd", 'he',          'shan', 'themselves', 'me',
               'out',             'and',        'other', 'because',         'will', 'it', 
'yourselves', 'or', 'should', 'y',              'both', 'during', 'on', 'all', 'who', 'below',
           "she's", 'our', 'once', 'ours', 'does', 'the',           'we', 'from', 'shouldn', 'its',
'mightn', 'whom', 'itself', 'after',                'few', 'ma', 'll', 'but',           'by', 'any',
'hers', 'to', 'now',            'have', 'each', 'down', 'this', 'some', "that'll", 'himself', 'hadn',
'd', 'a', 'than', 'am',            'with', 'my', 'most', 'm', 'herself', 'under', 'these', "hasn't",
'for', 'off', 'more', "it's", 'too', 'won', "you're", 'before', 'ourselves',         'here', 'is'}

def naive(tr,te,we):
    train = pd.read_csv(tr).values
    xtest = pd.read_csv(te).values
    # print(train.shape,xtest.shape)
    
    # x = [[ps.stem(word) for  word in re.split('[^A-Za-z0-9\']+',row) if word not in stop] for row in train[:,0]]
    x = [[word for  word in re.split('[^A-Za-z0-9\']+',row) if word not in stop] for row in train[:,0]]
    y = [[x[i][j]+' '+x[i][j+1] for j in range(len(x[i])-1)] for i in range(len(x))]
    temp = [x[i].extend(y[i]) for i in range(len(x))]
    z = [x[i] for i in range(len(x))] 
    
    allWords = set()
    A = {}
    p=0
    totPosWords = 0
    totNegWords = 0
    
    for i in range(len(z)):
        if(train[i,1]=='positive'):
            p += 1
        for word in z[i]:
            if word.lower() in allWords:
                if train[i,1] == 'positive':
                    totPosWords += 1 
                    A[word.lower()] = (A[word.lower()][0]+1,A[word.lower()][1])
                else:
                    totNegWords += 1
                    A[word.lower()] = (A[word.lower()][0],A[word.lower()][1]+1)
            else:
                allWords.add((word.lower()))
                A[word.lower()] = (0,0)
    
    prob_pos = p/train.shape[0]
    prob_neg = 1-prob_pos
    # print(prob_pos)
    # print(totPosWords,totNegWords)
   
    del(x)
    del(y)
    del(temp)
    del(z)
    # x = [[ps.stem(word) for  word in re.split('[^A-Za-z0-9\']+',row) if word not in stop] for row in xtest[:,0]]
    x = [[word for  word in re.split('[^A-Za-z0-9\']+',row) if word not in stop] for row in xtest[:,0]]
    y = [[x[i][j]+' '+x[i][j+1] for j in range(len(x[i])-1)] for i in range(len(x))]
    temp = [x[i].extend(y[i]) for i in range(len(x))]
    z = [x[i] for i in range(len(x))]  
    
    denominatorP = totPosWords+len(allWords)
    denominatorN = totNegWords+len(allWords)
                
    i = 0
    for review in z:
        P = np.log(prob_pos)
        N = np.log(prob_neg)
        for word in review:
            tempP = 0
            tempN = 0
            if (word.lower() in allWords):
                tempP = A[word.lower()][0]
                tempN = A[word.lower()][1]
            P += np.log((1+tempP)/denominatorP)
            N += np.log((1+tempN)/denominatorN)
        if P > N:
            print(1,file=open(we,"a"))
        else:
            print(0,file=open(we,"a"))

naive(*sys.argv[1:])