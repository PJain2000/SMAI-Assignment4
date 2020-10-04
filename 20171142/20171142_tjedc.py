#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator


# In[2]:


df1 = pd.read_csv('./Datasets_final/q2/train.csv',dtype=str)
df2 = pd.read_csv('./Datasets_final/q2/test.csv',dtype=str)


# In[3]:


def append_ext(fn):
    return fn+".jpg"


# In[4]:


df1['image_file'] = df1['image_file'].apply(append_ext)
df2['image_file'] = df2['image_file'].apply(append_ext)


# # Create train_datagen and test_datagen

# In[5]:


# this is the augmentation configuration we will use for training
datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.25)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)


# # Initialize paths where images flow from

# In[6]:


train_generator = datagen.flow_from_dataframe(
        dataframe = df1,
        directory = './Datasets_final/q2/train',
        x_col = 'image_file',
        y_col = 'emotion',
        subset="training",
        batch_size = 16,
        class_mode = 'categorical',
        target_size = (150, 150))

val_generator = datagen.flow_from_dataframe(
        dataframe = df1,
        directory = './Datasets_final/q2/train',
        x_col = 'image_file',
        y_col = 'emotion',
        subset="validation",
        batch_size = 16,
        class_mode = 'categorical',
        target_size = (150, 150))

test_generator = test_datagen.flow_from_dataframe(
        dataframe = df2,
        directory = './Datasets_final/q2/test',
        x_col = 'image_file',
        y_col = None,
        batch_size = 16,
        class_mode = None,
        target_size = (150, 150))


# # Inializing the CNN
# 

# In[7]:


classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (150, 150, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(activation = 'relu', units = 128))
classifier.add(Dense(activation = 'sigmoid', units = 5))
classifier.compile (optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[8]:


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=val_generator.n//val_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

classifier.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=val_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=40
)


# In[11]:


classifier.evaluate_generator(generator=val_generator, steps=STEP_SIZE_TEST)


# In[12]:


pred=classifier.predict_generator(test_generator, steps=STEP_SIZE_TEST+1, verbose=1)


# In[13]:


predicted_class_indices=np.argmax(pred,axis=1)


# In[15]:


pd.DataFrame(predicted_class_indices, columns=['emotion']).to_csv("./Datasets_final/q2/final_submission2.csv", index=False)


# In[ ]:




