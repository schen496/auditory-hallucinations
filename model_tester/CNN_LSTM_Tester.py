from __future__ import print_function
import os
import numpy as np
from keras.models import load_model
import h5py
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

USE_TITANX=True

#############
### READING THE DATASET
# Define the external SSD where the dataset resides in
audio_prefix = 'seq3a'
if USE_TITANX:
    data_dir = '/home/zanoi/ZANOI/auditory_hallucinations_data/'
else:
    data_dir = '/Volumes/SAMSUNG_SSD_256GB/ADV_CV/data/'
file_name = data_dir + audio_prefix + '_dataX_dataY.h5'

# Open the h5py file
with h5py.File(file_name, 'r') as hf:
    print("Reading data from file..")
    dataX = hf['dataX'][:]
    dataY = hf['dataY'][:]
print("dataX.shape:", dataX.shape)
print("dataY.shape:", dataY.shape)

print("np.max(dataY)", np.max(dataY))
print("np.min(dataY)", np.min(dataY))

# Flatten out the data
print ("Flatenning data")
(num_videos, num_frames, timesteps, features) = dataX.shape  # (1,8377,1,4096)
dataX = np.reshape(dataX,(-1, timesteps, features))  # (8377,1,4096)
audio_vector_dim = dataY.shape[2]
dataY = np.reshape(dataY,(-1, audio_vector_dim))  # (8377,18)
print("dataX.shape:", dataX.shape)
print("dataY.shape:", dataY.shape)

def loadModel():
    ### Query user for the desired model
    dir = '../trained_models'

    files = [fname for fname in os.listdir(dir)
             if fname.endswith('.h5')]

    print("Hello, I found %s models in the directory '%s':" % (str(len(files)), dir))
    for i, file in enumerate(files):
        print("[%d]" % i, file)
    #model_idx = input("Select index [i] of desired model: ")
    model_idx = 0  # I'm just testing one model right now
    model_name = files[int(model_idx)]
    model_dir = dir + "/" + model_name

    print("Loading weights from: " + model_dir)
    model = load_model(model_dir)
    print("Model loaded successfully")
    return model

model = loadModel()

trainPredict = model.predict(dataX)
print ("trainPredict.shape", trainPredict.shape)
print ("dataY.shape", dataY.shape)
trainScore = math.sqrt(mean_squared_error(dataY[300:600,:], trainPredict[300:600,:]))
print('Train score: %.3f RMSE' % (trainScore))

trainPlot = model.predict(dataX[300:600,:])
print(trainPlot.shape)
plt.subplot(3,1,1)
plt.imshow(trainPlot.T, aspect='auto')
plt.subplot(3,1,2)
plt.imshow(dataY[300:600,:].T, aspect='auto')
plt.subplot(3,1,3)
plt.imshow(dataY[300:600,:].T, aspect='auto')
plt.colorbar()
plt.draw()
plt.savefig('CNN_LSTM-prediction.png')
plt.show()





