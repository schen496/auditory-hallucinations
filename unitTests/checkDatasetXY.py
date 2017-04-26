import h5py

# 25/5/2017
# This piece of code reads the contents of the generated dataset h5py file to make sure all the dimensions are correct

USE_TITANX = False

if USE_TITANX:
    data_dir = '/home/zanoi/ZANOI/auditory_hallucinations_data/TopAngle_data/'
else:
    data_dir = '/Volumes/SAMSUNG_SSD_256GB/ADV_CV/data/'

data_file = data_dir + 'TopAngleFinal_dataX_dataY.h5'

with h5py.File(data_file, 'r') as hf:
    print("Reading data from file..")
    dataX_sample = hf['dataX_train'][0]
    dataY_sample = hf['dataY_train'][0]
    print("dataX_sample.shape:", dataX_sample.shape)
    print("dataY_sample.shape:", dataY_sample.shape)

with h5py.File(data_file, 'r') as hf:
    dataX_train = hf['dataX_train']
    dataY_train = hf['dataY_train']
    dataX_test = hf['dataX_test']
    dataY_test = hf['dataY_test']
    print("dataX_train.shape:", dataX_train.shape)
    print("dataY_train.shape:", dataY_train.shape)
    print("dataX_test.shape:", dataX_test.shape)
    print("dataY_test.shape:", dataY_test.shape)




