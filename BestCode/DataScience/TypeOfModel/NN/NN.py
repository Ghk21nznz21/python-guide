import pickle
import tensorflow as tf
import keras_tuner as kt
from sklearn.preprocessing import StandardScaler
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, TensorBoard)
from tensorflow.keras.metrics import SparseCategoricalAccuracy  # Import the accuracy metric

from tensorflow.keras.layers import Dense

from datetime import datetime  # Add this import


class NNClass:
    
    def __init__(self):
        self.gpu = None
        self.compile_params = None
        self.model = None

    def setGPU(self):
        # Use a fraction of the GPU to enable multiple models at the same time.
        config = ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.333
        session = InteractiveSession(config=config)
        return

    def setCallbacks(self, monitor='val_accuracy', early_stop_patience=10):
        # Callback Early Stop
        early_stop = EarlyStopping(monitor=monitor, patience=early_stop_patience)
        
        # Callback Tensorboard for graphing results
        name = f'Model-{datetime.now()}'
        tensor_board = TensorBoard(log_dir=f'logs/{name}')
        
        # Callback Save Best Model, based on metric
        model_checkpoint_callback = ModelCheckpoint(
            filepath='checkpoint',
            save_weights_only=True,
            monitor=monitor,
            mode='max',
            save_best_only=True
        )
        self.callbacks = [early_stop, tensor_board, model_checkpoint_callback]
        return self.callbacks

    def setCompile(self, optimizer='ADAM', learning_rate=0.1, loss='MSE', metrics=None):
        if metrics is None:
            metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
        self.compile_params = {
            'optimizer': optimizers.get(optimizer)(learning_rate=learning_rate),
            'loss': loss,
            'metrics': metrics
        }

    def createModel(self, X, params: dict):
        if not params.get('activation'):
            params['activation'] = 'relu'

        # Model
        model = Sequential()
        # Input and 1st hidden layer
        model.add(Dense(params.get('nodes'), input_shape=(X.shape[1],)))
        
        # Hidden layers
        for _ in range(params.get('layers') - 1):
            model.add(Dense(params.get('nodes'), activation=params.get('activation'), 
                            kernel_regularizer=params.get('kernel_regularizer')))
        # Final Layer
        model.add(Dense(params.get('output_nodes'), activation='softmax'))
        # Optimizer, Learning Rate, Loss, Metrics
        model.compile(**self.compile_params)
        return model

    def trainModel(self, X, Y, model, batch_size=32, epochs=100, validation_split=0.15):
        # Usage of GPU (self.gpu)

        # Evaluation Graph
        history = model.fit(
            X, Y, batch_size=batch_size, epochs=epochs, 
            validation_split=validation_split, callbacks=self.callbacks
        )

        # Best Results
        val_acc_per_epoch = history.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print(f'Best epoch: {best_epoch}; Acc: {val_acc_per_epoch}')
        return model, history
    

    def hyperparameterTuning(self, X, Y, params: kt.HyperParameters, max_epochs=100, 
                             factor=3, max_trials=3):
        '''Auto Tunning with Keras
        params ex: kt.HyperParameters(hp_nodes = hp.Int('units', min_value=32, max_value=512, step=32))
        '''
    
        # Auto Tunning 
        tuner = kt.Hyperband(
            hypermodel=self.createModel(X, params, ),
            objective='val_accuracy',
            max_epochs=max_epochs,
            factor=factor,
            max_trials=max_trials,
            directory='tunning',
            project_name='hyper'
        )
        tuner.search(X, Y, epochs=50, validation_split=0.15, callbacks=self.callbacks)

        # Best Model
        best_model = tuner.get_best_models()[0]
        return best_model

    def customMLP(self, X, Y, args_dict):
        # Prepare validation metric
        acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

        # Prepare train batches
        train_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
        train_dataset = train_dataset.batch(args_dict['batch'])

        # Prepare the validation batches
        test_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
        test_dataset = test_dataset.batch(args_dict['batch'])

        # Loop epochs
        for epoch in range(args_dict['epoch']):
            print(f"\nStart of epoch {epoch}")
            # Loop batches
            for step, (x_batch, y_batch) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    y_pred = self.model(x_batch, training=True)
                    loss_value = self.custom_loss_formula(y_batch, y_pred)
                gradients = tape.gradient(loss_value, self.model.trainable_weights)
                max_gradients = loss_value * -1
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
                acc_metric.update_state(y_batch, y_pred)
                print(f'Accuracy for epoch {acc_metric.result()}')
                acc_metric.reset_states()

            for x_batch_val, y_batch_val in test_dataset:
                pred = self.model(x_batch_val, training=False)
                acc_metric.update_state(y_batch_val, pred)
            print(f'Validation result: {acc_metric.result()}')
            acc_metric.reset_states()