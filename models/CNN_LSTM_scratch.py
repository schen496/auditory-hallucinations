import scipy.io as sio
import os
import numpy as np
from collections import deque
import pickle
from keras.models import save_model
from keras.initializers import RandomNormal
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import h5py

# This is a CNN+LSTM intended to train from scratch directly from the video frames

USE_TITANX = True


def createModel(image_dim, audio_vector_dim):
    (img_rows, img_cols, img_channels) = image_dim  # (224,224,3)
    input_img = Input(shape=(img_rows, img_cols, img_channels))

    # Like Hanoi's work with DeepMind Reinforcement Learning, build a model that does not use pooling layers
    # to retain sensitivty to locations of objects
    x = Conv2D(filters=64,
               kernel_size=(16, 16),
               input_shape=image_dim,
               name='Input_Layer',
               activation='relu',
               padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
               strides=(8, 8))(input_img)

    x = Conv2D(filters=128,
               kernel_size=(8, 8),
               activation='relu',
               padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
               strides=(4, 4))(x)

    x = Conv2D(filters=256,
               kernel_size=(4, 4),
               activation='relu',
               padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
               strides=(2, 2))(x)

    x = Conv2D(filters=512,
               kernel_size=(2, 2),
               input_shape=image_dim,
               activation='relu',
               padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
               strides=(1, 1))(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu', name='1st_FC')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', name='2nd_FC')(x)
    # x = TimeDistributedDense(1)(x)

    # Note that LSTM expects input shape: (nb_samples, timesteps, feature_dim)
    x = Reshape((1, 512))(x)
    x = LSTM(256, input_shape=(1, 512), dropout=0.2, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(256, dropout=0.2, name='LSTM_reg_output')(x)
    network_output = Dense(audio_vector_dim)(x)

    model = Model(inputs=input_img, outputs=network_output)

    # Use the Adam optimizer for gradient descent
    adam = Adam(lr=1.5e-6)

    model.compile(loss='mean_squared_error', validation_split=0.1, optimizer='adam')
    print(model.summary())

    return model


# Testing if the model compiles
model = createModel((224, 224, 3), 18)

############
### LOADING AUDIO VECTORS ###
audio_feature_dir = "../audio_vectors"

audio_f_files = [os.path.join(audio_feature_dir, file_i)
                 for file_i in os.listdir(audio_feature_dir)
                 if file_i.endswith('.mat')]

num_audio_f = len(audio_f_files)
print("num_audio_f: ", num_audio_f)

###########
'''
### READING AUDIO VECTORS
audio_idx = 1 # 3 is the single seq3a one
audio_f_file = audio_f_files[audio_idx]  # Test with just one audio feature vector, and find all the corresponding movies
mat_contents = sio.loadmat(audio_f_file)  # 18 x n-2
audio_vectors = mat_contents['audio_vectors']
audio_vector_length = audio_vectors.shape[1]
#print(audio_f_files[0])
print("audio_vectors.shape: ", audio_vectors.shape)

# Extract the file prefix using regular expressions
start = audio_f_file.find('seq')
end = audio_f_file.find("_audio", start)
audio_prefix = audio_f_file[start:end]
'''

#############
### READING THE DATASET
# Define the external SSD where the dataset residesin
if USE_TITANX:
    data_dir = '/home/zanoi/ZANOI/auditory_hallucinations_data/'
else:
    data_dir = '/Volumes/SAMSUNG_SSD_256GB/ADV_CV/data/'

top_angles = ['seq1_angle5',
              'seq2_angle5',
              'seq3_angle4',]

data_files = [os.path.join(data_dir, file_i)
              for file_i in top_angles]

'''
for data_file in data_files:
    print ("data_file:", data_file)
'''

# Open the h5py file
dataX = []
dataY = []
for data_file in data_files:
    with h5py.File(data_file, 'r') as hf:
        print("Reading data from file..")
        dataX_vid = hf['dataX'][:]
        dataY_features = hf['dataY'][:]
    print("dataX_vid.shape:", dataX_vid.shape)
    print("dataY_features.shape:", dataY_features.shape)
    dataX.append(dataX_vid)
    dataY.append(dataY_features)

# Flatten out the data
print("Flatenning data")
(num_videos, num_frames, timesteps, features) = dataX.shape  # (1,8377,1,4096)
dataX = np.reshape(dataX, (-1, timesteps, features))  # (8377,1,4096)
audio_vector_dim = dataY.shape[2]
dataY = np.reshape(dataY, (-1, audio_vector_dim))  # (8377,18)
print("dataX.shape:", dataX.shape)
print("dataY.shape:", dataY.shape)

#############
### BUILD THE MODEL
model = createModel((timesteps, features), audio_vector_dim)

#############
### CHECK FOR EXISTING MODEL
model_name = ''
if os.path.exists("./checkpoints/" + model_name):
    model.load_weights("./checkpoints/" + model_name)

#############
### BEGIN TRAINING THE MODEL
# Set up Keras checkpoints to monitor the accuracy and save the model when it improves
# filepath="./checkpoints/CNN_LSTM-{epoch:02d}-{mae:.2f}.hdf5"
# checkpointCallBack = ModelCheckpoint(filepath, monitor='mae', verbose=1, save_best_only=True, mode='max')

# Setup tensorboard
# tbCallBack = TensorBoard(log_dir='./graph', histogram_freq=1, write_graph=True, write_images=True)

# Put these in a callback list
# callbacks_list = [checkpointCallBack, tbCallBack]

# This function actually starts the training
# model.fit(dataX, dataY, epochs=500, batch_size=256, callbacks=callbacks_list, verbose=2)
model.fit(dataX, dataY, epochs=500, batch_size=256, verbose=2)

print("Saving trained model...")
model_prefix = 'CNN_LSTM_v1'
model_path = "../trained_models/" + model_prefix + ".h5"
save_model(model, model_path, overwrite=True)  # saves weights, network topology and optimizer state (if any)

print("--- {EVERYTHING COMPLETE HOMIEEEEEEEEE} ---")
