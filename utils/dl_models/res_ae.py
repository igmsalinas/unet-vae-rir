"""
Ignacio Martín 2022

Code corresponding to the Bachelor Thesis "Synthesis of Room Impulse Responses by Means of Deep Learning"

Authors:
    Ignacio Martín
    José Antonio Belloch
    Gema Piñero

University Carlos III de Madrid

Credit to Valerio Velardo for the general architecture of the model, available at:
https://github.com/musikalkemist

"""

import os
import pathlib
import pickle
import numpy as np
import tensorflow as tf

from keras.layers import LeakyReLU
from keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, \
    Flatten, Dense, Reshape, Conv2DTranspose, Activation, concatenate, Embedding, Dropout, Add

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError


class ResAE:
    """
    The Residual Autoencoder class represents a Deep Convolutional Residual Autoencoder with
    mirrored encoder and decoder with the addition of the information vector as an input.
    """

    def __init__(self,
                 input_shape,
                 inf_vector_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim,
                 n_neurons,
                 name="ResAE"
                 ):
        self.input_shape = input_shape  # [128, 144, 2]
        self.inf_vector_shape = inf_vector_shape  # [10, 2]
        self.conv_filters = conv_filters  # [2, 4, 8]
        self.conv_kernels = conv_kernels  # [3, 5, 3]
        self.conv_strides = conv_strides  # [1, 2, 2]
        self.latent_space_dim = latent_space_dim  # 2
        self.n_neurons = n_neurons
        self.name = name

        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build()

    def summary(self):
        """
        Describes the models' architecture and layers connections alongside number of parameters
        """
        self.encoder.summary()
        self.decoder.summary()
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
                        y_val, batch_size, num_epochs, steps_per_epoch,
                        learning_rate=0.00001):
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
            learning_rate,
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
        latent_representations = self.encoder.predict(inputs)
        reconstructed_stft = self.decoder.predict(latent_representations)
        return reconstructed_stft

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
        autoencoder = ResAE(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder

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
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim,
            self.n_neurons
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
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self):
        """
        Builds the Autoencoder by associating the input
        and the relationship between the decoder and the encoder
        """
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="autoencoder")

    def _build_decoder(self):
        """
        Builds the Decoder
        """
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        first_conv = self._add_first_conv(reshape_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(first_conv)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        """
        Adds the decoder input

        :return: tf.keras.layer
        """
        return Input(shape=self.latent_space_dim, name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        """
        Adds Dense layer to decoder input

        :param decoder_input: Decoder input to apply Dense layer
        :return: tf.keras.layer
        """
        num_neurons = np.prod(self._shape_before_bottleneck)  # [1, 2, 4] -> 8
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        output = Dropout(.3)(dense_layer)
        return output

    def _add_reshape_layer(self, dense_layer):
        """
        Reshapes the dense layer into the one before the bottleneck for reconstruction

        :param dense_layer: Dense layer input
        :return: tf.keras.layer
        """
        return Reshape(self._shape_before_bottleneck)(dense_layer)

    def _add_first_conv(self, x):
        """
        Adds the first residual deconvolution block after Reshaping

        :param x: Reshaped Dense layer
        :return: tf.keras.layer
        """
        name = "d_res_0"
        x = self.res_t_conv(x, strides=1, filters=self.conv_filters[-1], kernels=self.conv_kernels[-1], name=name)
        x = self.res_t_identity(x, filters=self.conv_filters[-1], kernels=self.conv_kernels[-1], name=name)
        return x

    def _add_conv_transpose_layers(self, x):
        """
        Adds all residual deconvolution blocks.

        :param x: First deconvolution
        :return tf.keras.layer
        """
        # loop through all the conv layers in reverse order and stop at the
        # first layer
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        """
        Adds a residual deconvolution block consisting of a residual deconvolution followed
        by a residual identity

        :param layer_index: Index of the deconvolution layer
        :param x: Previous layer
        :return: tf.keras.layer
        """
        layer_num = self._num_conv_layers - layer_index
        name = f"d_res_{layer_num}"
        x = self.res_t_conv(x, self.conv_strides[layer_index-1], filters=self.conv_filters[layer_index-1],
                            kernels=self.conv_kernels[layer_index], name=name)
        x = self.res_t_identity(x, filters=self.conv_filters[layer_index-1],
                                kernels=self.conv_kernels[layer_index], name=name)
        return x

    @staticmethod
    def res_t_identity(x, filters, kernels, name):
        """
        Residual Identity block consisting on 3 x Conv2DTranspose, BN, LeakyReLU

        :param x: Input layer
        :param filters: Number of filters in Conv2DTranspose
        :param kernels: Kernel size in Conv2DTranspose
        :param name: Name of block
        :return: tf.keras.layer Block
        """
        x_skip = x  # this will be used for addition with the residual block
        f = filters
        k = kernels
        x = Conv2DTranspose(f, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001),
                            name=name + "_id.1")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2DTranspose(f, kernel_size=(k, k), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001),
                            name=name + "_id.2")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2DTranspose(f, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001),
                            name=name + "_id.3")(x)
        x = BatchNormalization()(x)
        x = Add()([x, x_skip])
        x = LeakyReLU()(x)
        return x

    @staticmethod
    def res_t_conv(x, strides, filters, kernels, name):
        """
        Residual Convolution block consisting on 3 x Conv2DTranspose, BN, LeakyReLU

        :param x: Input layer
        :param strides: Strides in Conv2DTranspose
        :param filters: Number of filters in Conv2DTranspose
        :param kernels: Kernel size in Conv2DTranspose
        :param name: Name of block
        :return: tf.keras.layer Block
        """
        x_skip = x
        f = filters
        k = kernels
        s = strides
        x = Conv2DTranspose(f, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001),
                            name=name + "_conv.1")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2DTranspose(f, kernel_size=(k, k), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001),
                            name=name + "_conv.2")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2DTranspose(f, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001),
                            name=name + "_conv.3")(x)
        x = BatchNormalization()(x)
        x_skip = Conv2DTranspose(f, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001),
                                 name=name + "_conv.s")(x_skip)
        x_skip = BatchNormalization()(x_skip)
        x = Add()([x, x_skip])
        x = LeakyReLU()(x)
        return x

    def _add_decoder_output(self, x):
        """
        Adds the decoder output

        :param x: Last deconvolution layer
        :return: Sigmoid output
        """
        conv_transpose_layer = Conv2DTranspose(
            filters=2,
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"d_out_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer

    def _build_encoder(self):
        """
        Builds the Encoder
        """
        encoder_input1, encoder_input_inf = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input1)
        dense_inf = self._add_dense_to_inf(encoder_input_inf)
        bottleneck = self._add_bottleneck(conv_layers, dense_inf)
        self._model_input = [encoder_input1, encoder_input_inf]
        self.encoder = Model(self._model_input, bottleneck, name="encoder")

    def _add_encoder_input(self):
        """
        Adds the encoder inputs

        :return: tf.keras.layer
        """
        return Input(shape=self.input_shape, name="e_in"), \
               Input(shape=self.inf_vector_shape, name="e_in_vec")

    def _add_dense_to_inf(self, encoder_inf_input):
        """
        Adds Embedding and Dense layer to inf_vector input

        :param encoder_inf_input: Inf_vector model input
        :return: tf.keras.layer
        """
        n_neurons = self.n_neurons
        features = Embedding(2000, 256)(encoder_inf_input)
        x = Flatten()(features)
        x = Dense(n_neurons, name="e_dense_vector")(x)
        return x

    def _add_conv_layers(self, encoder_input):
        """
        Adds all residual convolution blocks.

        :param x: First deconvolution
        :return tf.keras.layer
        """
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        """
        Adds a residual convolution block consisting of a residual convolution followed
        by a residual identity

        :param layer_index: Index of the deconvolution layer
        :param x: Previous layer
        :return: tf.keras.layer
        """
        layer_number = layer_index + 1
        name = f"e_res_{layer_number}"
        x = self.res_conv(x, strides=self.conv_strides[layer_index], filters=self.conv_filters[layer_index],
                          kernels=self.conv_kernels[layer_index], name=name)
        x = self.res_identity(x, filters=self.conv_filters[layer_index],
                              kernels=self.conv_kernels[layer_index], name=name)
        return x

    @staticmethod
    def res_identity(x, filters, kernels, name):
        """
        Residual Identity block consisting on 3 x Conv2D, BN, LeakyReLU

        :param x: Input layer
        :param filters: Number of filters in Conv2DTranspose
        :param kernels: Kernel size in Conv2DTranspose
        :param name: Name of block
        :return: tf.keras.layer Block
        """
        x_skip = x  # this will be used for addition with the residual block
        f1 = filters
        k = kernels
        x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001),
                   name=name + "_id.1")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(f1, kernel_size=(k, k), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001),
                   name=name + "_id.2")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001),
                   name=name + "_id.3")(x)
        x = BatchNormalization()(x)
        x = Add()([x, x_skip])
        x = LeakyReLU()(x)
        return x

    @staticmethod
    def res_conv(x, strides, filters, kernels, name):
        """
        Residual Convolution block consisting on 3 x Conv2D, BN, LeakyReLU

        :param x: Input layer
        :param strides: Strides in Conv2DTranspose
        :param filters: Number of filters in Conv2DTranspose
        :param kernels: Kernel size in Conv2DTranspose
        :param name: Name of block
        :return: tf.keras.layer Block
        """
        x_skip = x
        s = strides
        f = filters
        k = kernels
        x = Conv2D(f, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001),
                   name=name + "_conv.1")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(f, kernel_size=(k, k), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001),
                   name=name + "_conv.2")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(f, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001),
                   name=name + "_conv.3")(x)
        x = BatchNormalization()(x)
        x_skip = Conv2D(f, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001),
                        name=name + "_conv.s")(x_skip)
        x_skip = BatchNormalization()(x_skip)
        x = Add()([x, x_skip])
        x = LeakyReLU()(x)
        return x

    def _add_bottleneck(self, x, y):
        """
        Flattens the data, concatenates it and applies Dense (latent space)

        :param x: Last convolutional block output
        :param y: Dense output from inf_vector
        :return: tf.keras.layer
        """
        self._shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten()(x)
        y = Flatten()(y)
        x = concatenate([x, y])
        x = Dense(self.latent_space_dim, name="e_out")(x)
        x = Dropout(.3)(x)
        return x


if __name__ == "__main__":

    residual_ae = ResAE(
        input_shape=(144, 302, 2),
        inf_vector_shape=(2, 18),
        conv_filters=(64, 128, 256, 512),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2),
        latent_space_dim=64,
        n_neurons=64*64,
        name="ResAE"
    )
    residual_ae.summary()
