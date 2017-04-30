'''
Lamtharn (Hanoi) Hantrakul
29/4/2017

=====================
HYPERPARAMETER SEARCH
=====================
This is a controller for doing hyperparameter search over models used in the project.

'''

from training_pipeline.exp_models.CNN_LSTM_models import *  # Contains the models we are testing
from training_pipeline.lr_w_search.model_utils import *  # Contains some useful functions for loading datasets

#########################
#Set TITANX and data_name

USE_TITANX = True  # Set to True if using the Linux Machine with TitanX
data_name = 'TopAngle100_dataX_dataY.h5'  # Set to name of h5py dataset

"""
Main function call for doing random hyperparameter search
"""
def main():

    num_trials = 100
    num_epochs = 10
    final_accuracies = []
    learning_rates = []
    weight_scales = []

    for i in range(num_trials):
        # random search over logarithmic space of hyperparameters
        lr = 10**np.random.uniform(-3.0,-7.0)
        ws = 10**np.random.uniform(-2.0,-4.0)
        learning_rates.append(lr)
        weight_scales.append(ws)

        print('---------------------{learning rate: %8f | weight scale: %6f | trial %i/%i}--------------------------' % (lr, ws, i+1, num_trials))

        """Read a sample from h5py dataset and return key dimensions

            Example returns:
                frame_h = 100
                frame_w = 100
                channels = 3
                audio_vector_dim = 18
        """
        frame_h, frame_w, channels, audio_vector_dim = returnH5PYDatasetDims(USE_TITANX=USE_TITANX,
                                                                             data_name=data_name)

        """Load full dataset as an HDF5 matrix object for use in Keras model

                Example returns and corresponding matrix shapes:
                    dataX_train.shape = (26000,100,100,3)
                    dataY_train.shape = (26000,18)
                    dataX_train.shape = (4000,100,100,3)
                    dataY_train.shape = (4000,18)
        """
        dataX_train, dataY_train, dataX_test, dataY_test = load5hpyData(USE_TITANX=USE_TITANX,
                                                                        data_name=data_name)

        # create defined model with given hyper parameters
        model = CNN_LSTM_model(image_dim=(frame_h,frame_w,channels),
                               audio_vector_dim=audio_vector_dim,
                               learning_rate=lr,
                               weight_init=ws)

        # load custom callbacks
        loss_history = LossHistory()
        acc_history = AccuracyHistory()
        callbacks_list = [loss_history, acc_history]

        # train the model
        model.fit(dataX_train,
                  dataY_train,
                  shuffle='batch',
                  epochs=num_epochs,
                  batch_size=10000,  # 10000 is the maximum number of samples that fits on TITANX 12GB Memory
                  validation_data=(dataX_test, dataY_test),
                  verbose=1,
                  callbacks=callbacks_list)

        print ("--- Training Complete, plotting and saving training and accuracy history ---")
        # Graph training history
        plotAndSaveData(loss_history.train_losses, loss_history.test_losses, "Loss", learning_rate_hp=lr, weight_hp=ws, title="Loss History")
        plotAndSaveData(acc_history.train_acc, acc_history.test_acc, "Accuracy", learning_rate_hp=lr, weight_hp=ws, title="Acc History")
        print ("...Plotting complete!")

        # Save final accuracy
        final_accuracies.append(acc_history.test_acc[-1])

        print("--- Using model to predict spectrum for test set and saving result ---")
        # Make a directory to store the predicted spectrums
        folder_name = '{lr:%s}-{ws:%s}' % (str(lr), str(ws))
        dir_path = makeDir(folder_name)  # dir_path = "../graphs/predicted_spectrums/{lr:0.000597}-{ws:0.000759}"
        # Run the model on some unseen data and save the predicted spectrums in the directory defined previously
        genAndSavePredSpectrum(model,
                               dir_path,
                               window_length = 300,
                               USE_TITANX=USE_TITANX,
                               data_name=data_name)
    print("...Predictions complete!")

    #Once all trials are complete, make a 3D plot that graphs x: learning rate, y: weight scale and z: final_accuracy
    plotAndSaveSession(learning_rates,weight_scales,final_accuracies)

    print("--- {EVERYTHING COMPLETE HOMIEEEEEEEEE} ---")

if __name__ == '__main__':
    main()