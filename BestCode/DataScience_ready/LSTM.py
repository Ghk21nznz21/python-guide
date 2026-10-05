import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class LSTMPredictor:
    """ LSTM model for time series prediction """
    
    def __init__(self):
        ''' Initialize the LSTMPredictor with empty scalers and model '''
        self.train_scaler = []
        self.test_scaler = []
        self.model = None

    def normalize_data(self, data: list |pd.DataFrame, train: bool):
        """
        Normalize the data to a range of 0 to 1.
        If the data is a list or numpy array, it is normalized directly.
        If the data is a DataFrame, a MinMaxScaler is used.
        """
        if isinstance(data, (list, np.ndarray)):
            # 1D array 
            min_val = min(data)
            max_val = max(data)
            if train:
                self.train_scaler = [min_val, max_val]
            else:
                self.test_scaler = [min_val, max_val]
            return [(x - min_val) / (max_val - min_val) for x in data]
        if train:
            self.train_scaler = MinMaxScaler(feature_range=(0, 1))
            self.train_scaler.fit(data)
            return self.train_scaler.transform(data)
        else:
            self.test_scaler = MinMaxScaler(feature_range=(0, 1))
            self.test_scaler.fit(data)
            return self.test_scaler.transform(data)

    def create_sequences(self, data, sequence_length):
        """
        Create sequences of the given length from the data.
        Each sequence is used to predict the next value in the data.
        """
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i - sequence_length:i])
            y.append(data[i])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y
    
    def create_sequences_multipredictors(self, data, target, sequence_length):
        """
        Create sequences of the given length from the data.
        Each sequence is used to predict the target value.
        """
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i - sequence_length:i, :])
            y.append(data[i, target])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y

    def build_model(self, units, layers, dropout_rate=0.1, learning_rate=0.1, l2_penalty=0.1):
        """ Build the LSTM model with the given parameters """
        model = Sequential()
        model = Sequential()
        for _ in range(layers):
            model.add(LSTM(units, return_sequences=True, kernel_regularizer=l2(l2_penalty)))
            model.add(Dropout(dropout_rate))
        model.add(LSTM(units, kernel_regularizer=l2(l2_penalty)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1))
        
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
        self.model = model
        return model 

    def train_model(self, X_train, y_train, epochs, batch_size, plot=True):
        """
        Train the LSTM model on the given data.
        If plot is True, a plot of the training loss is shown.
        """
        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        history = self.model.fit(
            X_train, y_train, epochs=epochs, 
            batch_size=batch_size, verbose = 0,
            callbacks = [early_stop]
        )
        if plot:
            # Plot the training loss
            plt.plot(history.history['loss'])
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.show()
        print('Loss', history.history['loss'][-1])

    def invert(self, target, train: bool, multi_predictors=False):
        """
        Invert the normalization of the target data.
        If multi_predictors is True, a 2D inverse transform is used.
        """
        scaler = self.train_scaler if train else self.test_scaler
        
        if multi_predictors:
            # 2D
            target = scaler.inverse_transform(target)
        else:
            # 1D
            invert_transform = lambda x, min_val, max_val: x * (max_val - min_val) + min_val
            target = [invert_transform(x, scaler[0], scaler[1]) for x in target]
        return target
        
    
    def plot_predictions(self, X_test, y_train, y_test):
        ''' Plot the model's predictions against the actual values. '''
        y_test = self.invert(y_test, train=False)
        predictions = self.invert(self.model.predict(X_test), train=False)
        y_train = self.invert(y_train, train=True)
        
        # Test and Forecast plot 
        plt.plot(y_test, label='Valores Reais', marker='o')
        plt.plot(predictions, label='Previsões', linestyle='--')
        plt.title('Comparação entre Previsões e Valores Reais')
        plt.xlabel('Data')
        plt.ylabel('PJME')
        plt.legend()
        plt.show()
        
        # Train, Test and Forecast plot 
        x_test_range = range(len(y_train), len(y_train) + len(y_test))
        x_pred_range = range(len(y_train), len(y_train) + len(predictions))
        
        plt.figure(figsize=(12, 5), dpi=100)
        plt.plot(range(len(y_train)), y_train, label='training')
        plt.plot(x_test_range, y_test, label='actual')
        plt.plot(x_pred_range, predictions, label='forecast')
        plt.title('Comparação entre Previsões e Valores Reais')
        plt.legend(loc='upper left', fontsize=8)
        plt.show()
        
    def plot_predictions_recursively(self, starting_observations, y_test, y_train):
        '''
        First prediction made with know values. After that it starts using 
        predictions for future predictions
        '''
        predictions = []
        current_X = np.array(starting_observations).copy()
        for _ in range(len(y_test)):
            prediction = self.model.predict(current_X.reshape(1, -1), verbose=0)[0]    
            predictions.append(prediction)

            # use forecast 
            current_X = np.roll(current_X, shift=-1)  
            current_X[-1] = prediction


        # Invert the arrays
        y_test = self.invert(y_test, train=False)
        predictions = self.invert(np.array(predictions), train=False)
        y_train = self.invert(y_train, train=True)

        # Test and Forecast plot
        plt.plot(y_test, label='Valores Reais', marker='o')
        plt.plot(predictions, label='Previsões', linestyle='--')
        plt.title('Comparação entre Previsões e Valores Reais')
        plt.xlabel('Data')
        plt.ylabel('PJME')
        plt.legend()
        plt.show()

        # Train, Test and Forecast plot
        x_test_range = range(len(y_train), len(y_train) + len(y_test))
        x_pred_range = range(len(y_train), len(y_train) + len(predictions))

        plt.figure(figsize=(12, 5), dpi=100)
        plt.plot(range(len(y_train)), y_train, label='training')
        plt.plot(x_test_range, y_test, label='actual')
        plt.plot(x_pred_range, predictions, label='forecast')
        plt.title('Comparação entre Previsões e Valores Reais')
        plt.legend(loc='upper left', fontsize=8)
        plt.show()
        
        return predictions 



# Testing 
import numpy as np
import pandas as pd

# Create a sine wave data
x = np.linspace(0, 50, 1000)
y = np.sin(x)

# Convert to pandas DataFrame
data = pd.DataFrame(data=y, columns=["Sine"])

# Initialize the LSTMPredictor
predictor = LSTMPredictor()

# Normalize the data
normalized_data = predictor.normalize_data(data, train=True)

# Create sequences
sequence_length = 10
x, y = predictor.create_sequences(normalized_data, sequence_length)

# Build the model
predictor.build_model(units=50, layers=2, dropout_rate=0.2, learning_rate=0.001, l2_penalty=0.001)

# Train the model
predictor.train_model(x, y, epochs=100, batch_size=32, plot=True)

# Plot predictions
predictor.plot_predictions(x, y)