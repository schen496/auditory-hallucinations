from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from sklearn.metrics import mean_squared_error
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

class predictSeqCallback(Callback):
    def on_train_begin(self, logs={}):
        self.test_mean_errors = []
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        #y_pred = self.model.predict(self.model.validation_data[0])
        test_mean_error = predictSequence(epoch, self.model, 5000, self.model.validation_data[0], self.model.validation_data[1])
        self.test_mean_errors.append(test_mean_error)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def predictSequence(epoch_num, model, num_frames, dataX_test, dataY_test):
    window_length = 300
    num_windows = math.floor(num_frames / window_length)
    test_mean_errors = []
    for i in tqdm(range(num_windows)):
        pred_idx = i * window_length
        end_idx = pred_idx + window_length

        trainPredict = model.predict(dataX_test)
        trainScore = math.sqrt(mean_squared_error(dataY_test[pred_idx:end_idx, :], trainPredict[pred_idx:end_idx, :]))
        print('Train score: %.3f RMSE' % (trainScore))
        test_mean_errors.append(trainScore)

        ##### PLOT RESULTS
        trainPlot = model.predict(dataX_test[pred_idx:end_idx, :])
        print(trainPlot.shape)
        plt.subplot(3, 1, 1)
        plt.imshow(trainPlot.T, aspect='auto')
        plt.title('CNN-LSTM prediction / Overfit on Top Angles only')
        plt.ylabel('Note bins')
        plt.xlabel('Time (frames)')
        plt.subplot(3, 1, 2)
        plt.title('Ground Truth')
        plt.ylabel('Note bins')
        plt.xlabel('Time (frames)')
        plt.annotate('RMSE: %.3f' % (trainScore), xy=(5, 5), xytext=(5, 33))
        plt.imshow(dataY_test[pred_idx:end_idx, :].T, aspect='auto')
        plt.subplot(3, 1, 3)
        plt.imshow(dataY_test[pred_idx:end_idx, :].T, aspect='auto')
        plt.colorbar()
        plt.tight_layout()
        plt.draw()
        plt.savefig('../model_tester/CNN_LSTM_scratch_figures/' + str(epoch_num) + '_CNN_LSTM_scratch-pred_TopAngle' + str(i) + '.png')
        # plt.show()
        plt.close()

    return np.mean(test_mean_errors)