"""
Ignacio Martín 2022

Code corresponding to the Bachelor Thesis "Synthesis of Room Impulse Responses by Means of Deep Learning"

Authors:
    Ignacio Martín
    José Antonio Belloch
    Gema Piñero

University Carlos III de Madrid

Credit to Francisco Pastor for the general architecture of the model
"""

import os
import pathlib
import pickle
import numpy as np
import tensorflow as tf
from keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Reshape, Conv2DTranspose, Activation, Embedding, concatenate, \
    LeakyReLU, Dropout, Resizing

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError


class AENet:
    """
    Autoencoder represents a Deep Convolutional autoencoder architecture with
    mirrored encoder and decoder components with the addition of the information vector as an input.
    """

    def __init__(self, input_shape, inf_vector_shape,
                 learning_rate=1e-5,
                 mode=0, number_filters_0=32, BatchNorm=True,
                 resize_factor_0=None, res_factor=None,
                 name="AENet"
                 ):
        if res_factor is None:
            self.res_factor = [2, 2]
        if resize_factor_0 is None:
            self.resize_factor_0 = [1, 1]

        self.input_shape = input_shape  # Shape of the input tensor, the stacked spectrogram [128, 144, 2]
        self.inf_vector_shape = inf_vector_shape  # Shape of the information vector [10, 2]
        self.learning_rate = learning_rate
        self.mode = mode
        self.number_filters_0 = number_filters_0
        self.BatchNorm = BatchNorm
        self.name = name

        self.model = None

        self._model_input = None

        self._build()

    def summary(self):
        """
        Describes the models' architecture and layers connections alongside number of parameters
        """
        self.model.summary()

    def get_callbacks(self):
        """
        Method to obtain callbacks for the training

        :return: list of tf.keras.callbacks
        """
        return [
            tf.keras.callbacks.CSVLogger(f'{self.name}.log', separator=',', append=False),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20),
        ]

    def compile_and_fit(self, x_train1, x_train2, y_train, x_val1, x_val2,
                        y_val, batch_size, num_epochs, steps_per_epoch):
        """
        Fits the model to the training data

        :param x_train1: Training spectrogram
        :param x_train2: Training inf_vector
        :param y_train: Target spectrogram
        :param x_val1: Validation spectrogram
        :param x_val2: Validation inf_vector
        :param y_val: Validation target spectrogram
        :param batch_size: Batch size
        :param num_epochs: Max number of epochs
        :param steps_per_epoch: Steps per epoch
        :param learning_rate: Learning rate
        :return: History of training History.history
        """
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            self.learning_rate,
            decay_steps=steps_per_epoch * 100,
            decay_rate=1,
            staircase=False)
        optimizer = Adam(learning_rate=lr_schedule)
        mse_loss = MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=mse_loss)

        self.summary()

        history = self.model.fit(x=[x_train1, x_train2],
                                 y=y_train,
                                 validation_data=([x_val1, x_val2], y_val),
                                 batch_size=batch_size,
                                 epochs=num_epochs,
                                 callbacks=self.get_callbacks(),
                                 shuffle=False)
        return history.history

    def save(self, save_folder="."):
        """
        Saves the model parameters and weights

        :param save_folder: Directory for saving the data
        """
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def load_weights(self, weights_path):
        """
        Loads the pre-trained model weights

        :param weights_path: Directory of weights
        """
        self.model.load_weights(weights_path)

    def predict_stft(self, inputs):
        """
        Generates STFTs

        :param inputs: List of spectrograms and vectors to generate
        :return: Generated STFT
        """
        generated_stft = self.model.predict(inputs)
        return generated_stft

    @classmethod
    def load(cls, save_folder="."):
        """
        Loads a pre-trained model

        :param save_folder: Folder where the model is saved
        :return: tf.keras.Model
        """
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        ue = AENet(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        ue.load_weights(weights_path)
        return ue

    @staticmethod
    def _create_folder_if_it_doesnt_exist(folder):
        """
        Creates a directory

        :param folder: Directory to create
        """
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        """
        Saves parameters into selected folder

        :param save_folder: Folder to save
        """
        parameters = [
            self.input_shape,
            self.inf_vector_shape,
            self.learning_rate,
            self.mode,
            self.number_filters_0,
            self.BatchNorm
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        """
        Saves weights into selected folder

        :param save_folder: Folder to save
        """
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)

    def _build(self):
        """
        Builds tf.keras.Model
        """

        inputs = tf.keras.Input(self.input_shape)
        v_input = tf.keras.Input(self.inf_vector_shape)
        self._model_input = [inputs, v_input]

        # ----- ENCODING -----

        # Block 1, 1024 --> 512, Filters_0 x 1
        encoding_1_out = self.encoding_block(inputs, self.resize_factor_0, self.number_filters_0, 1, self.BatchNorm,
                                             mode_convolution=self.mode)
        # Block 2, 512 --> 256, Filters_0 x 2
        encoding_2_out = self.encoding_block(encoding_1_out, self.res_factor, self.number_filters_0, 2, self.BatchNorm,
                                             mode_convolution=self.mode)
        # Block 3, 256 --> 128, Filters_0 x 4
        encoding_3_out = self.encoding_block(encoding_2_out, self.res_factor, self.number_filters_0, 4, self.BatchNorm,
                                             mode_convolution=self.mode)
        # Block 4, 128 --> 64, Filters_0 x 8
        encoding_4_out = self.encoding_block(encoding_3_out, self.res_factor, self.number_filters_0, 8, self.BatchNorm,
                                             mode_convolution=self.mode)
        # Block 5, 64 --> 32, Filters_0 x 16
        encoding_5_out = self.encoding_block(encoding_4_out, self.res_factor, self.number_filters_0, 16, self.BatchNorm,
                                             mode_convolution=self.mode)

        vector_out = self.vector_block(v_input)
        encoding_out = Flatten()(encoding_5_out)
        latent_space = concatenate([encoding_out, vector_out])

        concat_out = self.recover_shape(latent_space, encoding_5_out.get_shape().as_list())

        # ----- DECODING -----

        # Block 2, 32 --> 64, Filters_0 x 8
        decoding_2_out = self.decoding_block(concat_out, encoding_4_out, self.res_factor, self.number_filters_0, 8,
                                             self.BatchNorm, mode_convolution=self.mode)
        # Block 3, 64 --> 128, Filters_0 x 4
        decoding_3_out = self.decoding_block(decoding_2_out, encoding_3_out, self.res_factor, self.number_filters_0, 4,
                                             self.BatchNorm, mode_convolution=self.mode)
        # Block 4, 128 --> 256, Filters_0 x 2
        decoding_4_out = self.decoding_block(decoding_3_out, encoding_2_out, self.res_factor, self.number_filters_0, 2,
                                             self.BatchNorm, mode_convolution=self.mode)
        # Block 5, 256 --> 512, Filters_0 x 1
        decoding_5_out = self.decoding_block(decoding_4_out, encoding_1_out, self.res_factor, self.number_filters_0, 1,
                                             self.BatchNorm, mode_convolution=self.mode)

        # ----- OUTPUT -------

        x = tf.keras.layers.UpSampling2D(size=(self.resize_factor_0[0], self.resize_factor_0[1]))(decoding_5_out)
        out = tf.keras.layers.Conv2D(2, (6, 6), padding='same')(x)
        output_layer = tf.keras.activations.relu(out, alpha=0.0, max_value=1)

        self.model = tf.keras.Model(self._model_input, output_layer, name="AENet")

    def recover_shape(self, latent, block_shape):
        shape = block_shape[1:]
        shape[2] = 2
        dim = np.prod(shape)
        x = Dense(dim)(latent)
        x = Dropout(.5)(x)
        x = Reshape(shape)(x)
        x = Conv2D(self.number_filters_0 * 16, (1, 1))(x)
        return x

    def vector_block(self, input_v):
        features = Embedding(2500, 256)(input_v)
        f_features = Flatten()(features)
        x = Dense(64*32)(f_features)
        x = Dropout(.5)(x)
        return x

    def encoding_block(self, input_layer, pooling_factor, number_filters_0, filters_factor, BatchNorm,
                       mode_convolution=1):
        # x = tf.keras.layers.AveragePooling2D(pool_size=(pooling_factor[0], pooling_factor[1]))(input_layer)  # pooling
        conv_layer = Conv2D(
            filters=number_filters_0 * filters_factor,
            kernel_size=2,
            strides=pooling_factor[0],
            padding="same",
            kernel_regularizer=l2(0.001)
        )
        x = conv_layer(input_layer)
        # x = self.convolutional_block_1(x, n_filters=number_filters_0 * filters_factor, BatchNorm=BatchNorm,
        #                                kernel_size=3)  # dimensionality normalization
        # Feature extraction block
        if self.mode == 0:
            x = self.convolutional_block_1(x, n_filters=number_filters_0 * filters_factor, BatchNorm=BatchNorm)
        elif self.mode == 1:
            x = self.convolutional_block_2(x, n_filters=number_filters_0 * filters_factor, BatchNorm=BatchNorm)
        elif self.mode == 2:
            x = self.residual_block_1(x, n_filters=number_filters_0 * filters_factor, BatchNorm=BatchNorm)
        elif self.mode == 3:
            x = self.residual_block_2(x, n_filters=number_filters_0 * filters_factor, BatchNorm=BatchNorm)

        return x

    def decoding_block(self, input_layer, skip_connection_layer, pooling_factor, number_filters_0, filters_factor,
                       BatchNorm,
                       mode_convolution=1):

        # Deconvolution
        # x = tf.keras.layers.UpSampling2D(size=(pooling_factor[0], pooling_factor[1]))(input_layer)
        conv_transpose_layer = Conv2DTranspose(
            filters=number_filters_0 * filters_factor,
            kernel_size=2,
            strides=pooling_factor[0],
            padding="same",
            kernel_regularizer=l2(0.001)
        )
        x = conv_transpose_layer(input_layer)
        # x = self.convolutional_block_1(x, n_filters=number_filters_0 * filters_factor, BatchNorm=BatchNorm,
        #                                kernel_size=3)
        # Skip connection and number of filters normalization
        x = tf.keras.layers.concatenate([skip_connection_layer, x])
        x = self.convolutional_block_1(x, n_filters=number_filters_0 * filters_factor, BatchNorm=BatchNorm,
                                       kernel_size=3)
        # Feature extraction block
        if self.mode == 0:
            x = self.convolutional_block_1(x, n_filters=number_filters_0 * filters_factor, BatchNorm=BatchNorm)
        elif self.mode == 1:
            x = self.convolutional_block_2(x, n_filters=number_filters_0 * filters_factor, BatchNorm=BatchNorm)
        elif self.mode == 2:
            x = self.residual_block_1(x, n_filters=number_filters_0 * filters_factor, BatchNorm=BatchNorm)
        elif self.mode == 3:
            x = self.residual_block_2(x, n_filters=number_filters_0 * filters_factor, BatchNorm=BatchNorm)

        return x

    # Blocks definition
    @staticmethod
    def residual_block_1(input_layer, n_filters, BatchNorm):

        x = tf.keras.layers.Conv2D(n_filters, 3, padding='same')(input_layer)
        if BatchNorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)

        x = tf.keras.layers.Conv2D(n_filters, 3, padding='same')(x)
        if BatchNorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)

        x = tf.keras.layers.Add()([x, input_layer])

        return x

    @staticmethod
    def residual_block_2(input_layer, n_filters, BatchNorm):

        x = tf.keras.layers.Conv2D(n_filters, 3, padding='same')(input_layer)
        if BatchNorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)

        x = tf.keras.layers.Conv2D(n_filters, 3, padding='same')(x)
        if BatchNorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)

        x2 = tf.keras.layers.Conv2D(n_filters, 3, padding='same')(input_layer)
        if BatchNorm:
            x2 = tf.keras.layers.BatchNormalization()(x2)
        x2 = tf.keras.layers.Activation(activation='relu')(x2)

        x = tf.keras.layers.Add()([x, x2])

        return x

    @staticmethod
    def convolutional_block_1(input_layer, n_filters, BatchNorm, kernel_size=3):

        x = tf.keras.layers.Conv2D(n_filters, kernel_size, padding='same')(input_layer)
        if BatchNorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)

        return x

    @staticmethod
    def convolutional_block_2(input_layer, n_filters, BatchNorm, stride=3):

        x = tf.keras.layers.Conv2D(n_filters, stride, padding='same')(input_layer)
        if BatchNorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)

        x = tf.keras.layers.Conv2D(n_filters, stride, padding='same')(x)
        if BatchNorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)

        return x

    @staticmethod
    def mse_coef(y_true, y_pred):

        y_true = tf.keras.backend.flatten(y_true)
        y_pred = tf.keras.backend.flatten(y_pred)

        loss = tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred))

        return loss

    @staticmethod
    def rmse_coef(y_true, y_pred):

        y_true = tf.keras.backend.flatten(y_true)
        y_pred = tf.keras.backend.flatten(y_pred)

        loss = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred)) + 1.0e-12)

        return loss

    @staticmethod
    def rmse_coef_slicing(y_true, y_pred):

        y_true = tf.slice(y_true, [0, 0, 20, 0], [32, 160, 12, 1])
        y_pred = tf.slice(y_pred, [0, 0, 20, 0], [32, 160, 12, 1])

        y_true = tf.keras.backend.flatten(y_true)
        y_pred = tf.keras.backend.flatten(y_pred)

        loss = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred)))

        return loss

    @staticmethod
    def l1_norm(y_true, y_pred):

        y_true = tf.keras.backend.flatten(y_true)
        y_pred = tf.keras.backend.flatten(y_pred)

        loss = tf.keras.backend.sum(tf.keras.backend.abs(y_true - y_pred))

        return loss


if __name__ == "__main__":
    aenet = AENet(input_shape=(144, 304, 2),
                  inf_vector_shape=(2, 16),
                  name="AENet"
                  )
    aenet.summary()
