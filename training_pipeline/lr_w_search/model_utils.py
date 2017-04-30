from training_pipeline.exp_models.CNN_LSTM_models import *
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import Callback
from keras.models import save_model


def getModel(model_name):
    """Get the correct model for training

    Args:
        model_name (string): name of the model type e.g "FC_LSTM"

    Returns:
        model (object): specified keras model

    """

    model = None
    if model_name == "FC_LSTM":
        model = 0
    elif model_name == "CNN_LSTM":
        model = CNN_LSTM_model()
    elif model_name == "CNN_LSTM_STATEFUL":
        model = 0

    return model


def load5hpyData(USE_TITANX=True, data_name='TopAngle100_dataX_dataY.h5'):
    """Load h5py data and return HDF5 object corresponding to X_train, Y_train, X_test, Y_test

        Args:
            USE_TITANX (boolean): set True if using the Linux Computer with TITANX
            data_name (string): name of the dataset e.g 'TopAngle100_dataX_dataY.h5'

        Returns:
            dataX_train (HDF5Matrix object): keras object for loading h5py datasets
            dataY_train (HDF5Matrix object): keras object for loading h5py datasets
            dataX_test (HDF5Matrix object): keras object for loading h5py datasets
            dataY_test (HDF5Matrix object): keras object for loading h5py datasets

        """

    if USE_TITANX:
        data_dir = '/home/zanoi/ZANOI/auditory_hallucinations_data/'
    else:
        data_dir = '/Volumes/SAMSUNG_SSD_256GB/ADV_CV/data/'

    data_file = data_dir + data_name  # data_name = 'TopAngle100_dataX_dataY.h5' by default

    # Load first element of data to extract information on video
    with h5py.File(data_file, 'r') as hf:
        print("Reading fukk data from file..")
        dataX_train = hf['dataX_train']  # Adding the [:] actually loads it into memory
        dataY_train = hf['dataY_train']
        dataX_test = hf['dataX_test']
        dataY_test = hf['dataY_test']
        print("dataX_train.shape:", dataX_train.shape)
        print("dataY_train.shape:", dataY_train.shape)
        print("dataX_test.shape:", dataX_test.shape)
        print("dataY_test.shape:", dataY_test.shape)

    # Load data into HDF5Matrix object, which reads the file from disk and does not put it into RAM
    dataX_train = HDF5Matrix(data_file, 'dataX_train')
    dataY_train = HDF5Matrix(data_file, 'dataY_train')
    dataX_test = HDF5Matrix(data_file, 'dataX_test')
    dataY_test = HDF5Matrix(data_file, 'dataY_test')

    return dataX_train, dataY_train, dataX_test, dataY_test


def returnH5PYDatasetDims(USE_TITANX=True, data_name='TopAngle100_dataX_dataY.h5'):
    """Load h5py data and return the dimensions of data in the dataet

            Args:
                USE_TITANX (boolean): set True if using the Linux Computer with TITANX
                data_name (string): name of the dataset e.g 'TopAngle100_dataX_dataY.h5'

            Returns:
                frame_h (int): image height
                frame_w (int): image width
                channels (int): number of channels in image
                audio_vector_dim (int): number of dimensions (or features) in audio vector

            """
    if USE_TITANX:
        data_dir = '/home/zanoi/ZANOI/auditory_hallucinations_data/'
    else:
        data_dir = '/Volumes/SAMSUNG_SSD_256GB/ADV_CV/data/'

    data_file = data_dir + data_name  # data_name = 'TopAngle100_dataX_dataY.h5' by default

    with h5py.File(data_file, 'r') as hf:
        print("Reading data sample from file..")
        dataX_sample = hf['dataX_train'][0]  # select one sample from (7233,244,244,3)
        dataY_sample = hf['dataY_train'][0]
        print("dataX_sample.shape:", dataX_sample.shape)
        print("dataY_sample.shape:", dataY_sample.shape)

    (frame_h, frame_w, channels) = dataX_sample.shape  # (224,224,3)
    audio_vector_dim = dataY_sample.shape[0]

    return frame_h, frame_w, channels, audio_vector_dim


class saveModelOnEpochEnd(Callback):
    """Custom callback for Keras which saves model on epoch end"""
    def on_epoch_end(self, epoch, logs={}):
        # Save the model at every epoch end
        print("Saving trained model...")
        model_prefix = 'CNN_LSTM'
        model_path = "../trained_models/" + model_prefix + ".h5"
        save_model(self.model, model_path,
                   overwrite=True)  # saves weights, network topology and optimizer state (if any)
        return

class LossHistory(Callback):
    """Custom callback for Keras which saves loss history of training and testing data"""
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.test_losses = []

    def on_batch_end(self, batch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.test_losses.append(logs.get('val_loss'))

class AccuracyHistory(Callback):
    """Custom callback for Keras which saves accuracy history of training and testing data"""
    def __init__(self):
        super().__init__()
        self.train_acc = []
        self.test_acc = []

    def on_batch_end(self, batch, logs={}):
        self.train_acc.append(logs.get('acc'))
        self.test_acc.append(logs.get('val_acc'))

def visualizeData(param_history):

    return 0