#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding


# In[2]:


df1 = pd.read_csv('./Datasets_final/q1/train.csv')
df2 = pd.read_csv('./Datasets_final/q1/test.csv')


# In[3]:


docs = df1['text'].values
labels = df1['labels'].values

x_val = df2['text'].values


# ## prepare tokenizer

# In[4]:


t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1


# ## integer encode the documents

# In[5]:


encoded_docs = t.texts_to_sequences(docs)
# print(encoded_docs)


# ## pad documents to a max length of 4 words

# In[6]:


max_length = 20
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# print(padded_docs)


# ## load the whole embedding into memory

# In[7]:


embeddings_index = dict()
f = open('./Datasets_final/q1/glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


# ## create a weight matrix for words in training docs

# In[8]:


embedding_matrix = zeros((vocab_size, 100))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# ## define model

# In[9]:


model = Sequential()
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=20, trainable=True)
model.add(e)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))


# ## compile the model

# In[10]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# ## summarize the model

# In[11]:


print(model.summary())


# ## fit the model

# In[12]:


model.fit(padded_docs, labels, epochs=1000, verbose=0)


# ## evaluate the model

# In[13]:


loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))


# In[14]:


encoded_docs_val = t.texts_to_sequences(x_val)
# print(encoded_docs_val)


# In[15]:


max_length = 20
padded_docs_val = pad_sequences(encoded_docs_val, maxlen=max_length, padding='post')
# print(padded_docs_val)


# In[16]:


y_pred = model.predict(padded_docs_val)


# In[17]:


y_pred1 = []
for i in y_pred:
    y_pred1.append(int(round(i[0])))


# In[18]:


y_pred1 = np.array(y_pred1)


# In[19]:


pd.DataFrame(y_pred1, columns=['labels']).to_csv("./Datasets_final/q1/final_submission.csv", index=True)


# In[ ]:




