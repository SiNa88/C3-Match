# ### Load data

# In[1]:

import time
import cv2
import glob
import pickle
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt


im = cv2.imread('Train/0/00000_00000_00000.png')
print(im.shape)


# In[2]:


# function to read and resize images, get labels and store them into np array
def get_image_label_resize(label, filelist, dim = (32, 32), dataset = 'Train'):
    x = np.array([cv2.resize(cv2.imread(fname), dim, interpolation = cv2.INTER_AREA) for fname in filelist])
    y = np.array([label] * len(filelist))

    #print('{} examples loaded for label {}'.format(x.shape[0], label))
    return (x, y)

# data for label 0. I store them in parent level so that they won't be uploaded to github
filelist = glob.glob('Train/'+'0'+'/*.png')
trainx, trainy = get_image_label_resize(0, glob.glob('Train/'+str(0)+'/*.png'))


# In[3]:

start = time.perf_counter()

# go through all other labels and store images into np array
for label in range(1, 43):
    filelist = glob.glob('Train/'+str(label)+'/*.png')
    x, y = get_image_label_resize(label, filelist)
    #print(x," ",y)
    trainx = np.concatenate((trainx ,x))
    trainy = np.concatenate((trainy ,y))
# save data into a pickle to later use
# trainx.dump('../trainx.npy')
# trainy.dump('../trainy.npy')


# In[11]:


# load data from pickle
#trainx = np.load('../Trainx.npy', allow_pickle=True)
#trainy = np.load('../Trainy.npy', allow_pickle=True)


# In[10]:


# get path for test images
testfile = pd.read_csv('Test.csv')['Path'].apply(lambda x: x).tolist()#apply(lambda x: '../' + x).tolist()

X_test = np.array([cv2.resize(cv2.imread(fname), (32, 32), interpolation = cv2.INTER_AREA) for fname in testfile])
# X_test.dump('../testx.npy')

y_test = np.array(pd.read_csv('Test.csv')['ClassId'])
# y_test.dump('../testy.npy')


# In[12]:


# load data from pickle
#X_test = np.load('../testx.npy', allow_pickle=True)
#y_test = np.load('../testy.npy', allow_pickle=True)


# In[13]:


# shuffle training data and split them into training and validation
indices = np.random.permutation(trainx.shape[0])
# 20% to val
split_idx = int(trainx.shape[0]*0.8)
train_idx, val_idx = indices[:split_idx], indices[split_idx:]
X_train, X_validation = trainx[train_idx,:], trainx[val_idx,:]
y_train, y_validation = trainy[train_idx], trainy[val_idx]


# In[14]:


# get overall stat of the whole dataset
n_train = X_train.shape[0]
n_validation = X_validation.shape[0]
n_test = X_test.shape[0]
image_shape = X_train[0].shape
n_classes = len(np.unique(y_train))
print("There are {} training examples ".format(n_train))
print("There are {} validation examples".format(n_validation))
print("There are {} testing examples".format(n_test))
print("Image data shape is {}".format(image_shape))
print("There are {} classes".format(n_classes))


# In[15]:


# convert the images to grayscale
X_train_gry = np.sum(X_train/3, axis=3, keepdims=True)
X_validation_gry = np.sum(X_validation/3, axis=3, keepdims=True)
X_test_gry = np.sum(X_test/3, axis=3, keepdims=True)

# Normalize data
X_train_normalized_gry = (X_train_gry-128)/128
X_validation_normalized_gry = (X_validation_gry-128)/128
X_test_normalized_gry = (X_test_gry-128)/128


# In[23]:


#get_ipython().run_line_magic('matplotlib', 'inline')
#plt.style.use('seaborn-colorblind')

# descriptions for each label
sign = pd.read_csv('signnames.csv')

# pick an image, display the original and the normalized gray image
index = np.random.randint(0, n_train)
#fig, ax = plt.subplots(1,2)
#ax[0].set_title('original ' + sign.loc[sign['ClassId'] ==y_train[index], 'SignName'].values[0])
#ax[0].imshow(cv2.cvtColor(X_train[index], cv2.COLOR_BGR2RGB))

#ax[1].set_title('norm_gry ' + sign.loc[sign['ClassId'] ==y_train[index], 'SignName'].values[0])
#ax[1].imshow(X_train_normalized_gry[index].squeeze(), cmap='gray')


# In[9]:


# update the train, val and test data with normalized gray images
X_train = X_train_normalized_gry
X_validation = X_validation_normalized_gry
X_test = X_test_normalized_gry


# In[78]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models

model = models.Sequential()
# Conv 32x32x1 => 28x28x6.
model.add(layers.Conv2D(filters = 6, kernel_size = (5, 5), strides=(1, 1), padding='valid',
                        activation='relu', data_format = 'channels_last', input_shape = (32, 32, 1)))
# Maxpool 28x28x6 => 14x14x6
model.add(layers.MaxPooling2D((2, 2)))
# Conv 14x14x6 => 10x10x16
model.add(layers.Conv2D(16, (5, 5), activation='relu'))
# Maxpool 10x10x16 => 5x5x16
model.add(layers.MaxPooling2D((2, 2)))
# Flatten 5x5x16 => 400
model.add(layers.Flatten())
# Fully connected 400 => 120
model.add(layers.Dense(120, activation='relu'))
# Fully connected 120 => 84
model.add(layers.Dense(84, activation='relu'))
# Dropout
model.add(layers.Dropout(0.2))
# Fully connected, output layer 84 => 43
model.add(layers.Dense(43, activation='softmax'))


# In[79]:


model.summary()


# In[80]:


# specify optimizer, loss function and metric
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# training batch_size=128, epochs=10
conv = model.fit(X_train, y_train, batch_size=128, epochs=10,
                    validation_data=(X_validation, y_validation))


# In[82]:


acc = [conv.history['accuracy'], conv.history['val_accuracy']]
loss = [conv.history['loss'], conv.history['val_loss']]

epoch = range(10)

#plt.figure(figsize=(12, 6))
#plt.subplot(1, 2, 1)
#plt.plot(epoch, acc[0], label='Training Accuracy')
#plt.plot(epoch, acc[1], label='Validation Accuracy')
#plt.legend(loc='lower right')
#plt.title('Training and Validation Accuracy')

#plt.subplot(1, 2, 2)
#plt.plot(epoch, loss[0], label='Training Loss')
#plt.plot(epoch, loss[1], label='Validation Loss')
#plt.legend(loc='upper right')
#plt.title('Training and Validation Loss')
#plt.show()


# In[54]:


model.evaluate(x=X_test, y=y_test)


# In[97]:


# Save the entire model to a HDF5 file.
model.save('traffic_sign_seq_highaccuracy.h5')


elapsedTrain = time.perf_counter() - start
print ('Train Time: %.3f seconds' % elapsedTrain)

index = np.random.randint(0, n_test)
im = X_test[index]
#fig, ax = plt.subplots()
#ax.set_title(sign.loc[sign['ClassId'] ==np.argmax(model.predict(np.array([im]))), 'SignName'].values[0])
#ax.imshow(im.squeeze(), cmap = 'gray')