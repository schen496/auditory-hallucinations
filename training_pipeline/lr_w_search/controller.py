'''
Lamtharn (Hanoi) Hantrakul
29/4/2017
This is a controller for doing hyperparameter search over models used in the project.

'''

from training_pipeline.exp_models.CNN_LSTM_models import *
from training_pipeline.lr_w_search.model_utils import *  # Contains some useful functions for loading datasets


USE_TITANX = True  # Set to True if using the Linux Machine with TitanX
data_name='TopAngle100_dataX_dataY.h5'  # Set to name of h5py dataset

"""Read a sample from h5py dataset and return key dimensions

    Example returns:
        frame_h = 100
        frame_w = 100
        channels = 3
        audio_vector_dim = 18
"""
frame_h, frame_w, channels, audio_vector_dim = returnH5PYDatasetDims(USE_TITANX=USE_TITANX,
                                                                     data_name=data_name)

# load full dataset as an HDF5 matrix object for use in Keras model
dataX_train, dataY_train, dataX_test, dataY_test = load5hpyData(USE_TITANX=USE_TITANX,
                                                                data_name=data_name)

# create defined model with given hyper parameters
model = CNN_LSTM_model(image_dim=(frame_h,frame_w,channels),
                       audio_vector_dim=audio_vector_dim,
                       learning_rate=0.4e-6,
                       weight_init=0.01)

# load custom callbacks
loss_history = LossHistory()
acc_history = AccuracyHistory()
callbacks_list = [loss_history, acc_history]

# train the model
model.fit(dataX_test,
          dataY_test,
          shuffle='batch',
          epochs=20,
          batch_size=10000,  # 10000 is the maximum number of samples that fits on TITANX 12GB Memory
          validation_data=(dataX_test, dataY_test),
          verbose=1,
          callbacks = callbacks_list)

# Graph training history
for param_history in callbacks_list:
    visualizeData(param_history=param_history)










'''
N_train = small_data['X_train'].shape[0]
N_val = small_data['X_val'].shape[0]

small_data_corr_dim = {
    'X_train': small_data['X_train'].reshape(N_train, -1), # training data
    'y_train': small_data['y_train'], # training labels
    'X_val': small_data['X_val'].reshape(N_val, -1), # validation data
    'y_val': small_data['y_val'], # validation labels
  }

#max_count = 100
#for count in xrange(max_count):
#lr = 10**np.random.uniform(-3.0,-4.0)
#ws = 10**np.random.uniform(-2.0,-3.0)
ws = 0.0031
lr = 0.00045

print "----------- { NEW PARAMETERS } ---------------"
print "weight_scale: ", ws, "  learning_rate ", lr
model = FullyConnectedNet([100, 100],
                          weight_scale=ws,
                          dtype=np.float64)
solver = Solver(model,
                small_data_corr_dim,
                print_every=10, num_epochs=20, batch_size=25,
                update_rule='sgd',
                optim_config={
                  'learning_rate': lr,
                }
         )
solver.train()
'''