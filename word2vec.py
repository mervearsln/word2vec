# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import nltk
# importing all necessary modules 

from nltk.corpus import stopwords

# importing all necessary modules
from nltk.tokenize import sent_tokenize, word_tokenize

import gensim
from gensim.models import Word2Vec


# Read file
sample=open("./LittleRedRidingHood.txt", "r")
s= sample.read()

# Replace escape character with space
f = s.replace("\n", " ")

data = []


# iterate through each sentence in the file
for i in sent_tokenize(f):
    temp = []
      
    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())
  
    data.append(temp)
  
    
# Create CBOW model
model1 = gensim.models.Word2Vec(data, min_count = 1, 
                              size = 100, window = 5)


# Print results
print("Cosine similarity between 'alice' " + 
               "and 'wonderland' - CBOW : ",
    model1.similarity('wolf', 'grandmother'))
      

  


