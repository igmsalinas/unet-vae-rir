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

from tensorflow.keras import Model
from keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Reshape, Conv2DTranspose, Activation, Embedding, concatenate, LeakyReLU, Dropout

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError


class Autoencoder:
    """
    Autoencoder represents a Deep Convolutional autoencoder architecture with
    mirrored encoder and decoder components with the addition of the information vector as an input.
    """

    def __init__(self, input_shape, inf_vector_shape,
                 conv_filters, conv_kernels, conv_strides,
                 latent_space_dim, n_neurons,
                 name="Autoencoder"
                 ):
        self.input_shape = input_shape  # Shape of the input tensor, the stacked spectrogram [128, 144, 2]
        self.inf_vector_shape = inf_vector_shape  # Shape of the information vector [10, 2]
        self.conv_filters = conv_filters  # Number of filters to be applied in each layer [32, 64, 128, 256]
        self.conv_kernels = conv_kernels  # Number of kernels per layer[3, 3, 3]
        self.conv_strides = conv_strides  # Number of strides used [2, 2, 2]
        self.latent_space_dim = latent_space_dim  # Dimension of the bottleneck 32
        self.n_neurons = n_neurons  # Number of neurons receiving the information vector
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
                                 shuffle=True)
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

        generated_stft = self.model(inputs)
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
        ae = Autoencoder(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        ae.load_weights(weights_path)
        return ae

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
        Adds the first deconvolution block after Reshaping

        :param x: Reshaped Dense layer
        :return: tf.keras.layer
        """
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[-1],
            kernel_size=self.conv_kernels[-1],
            strides=1,
            padding="same",
            kernel_regularizer=l2(0.001),
            name="decoder_conv_transpose_layer_0"
        )
        x = conv_transpose_layer(x)
        x = BatchNormalization(name=f"decoder_bn_0")(x)
        x = ReLU(name=f"decoder_leakyrelu_0")(x)
        return x

    def _add_conv_transpose_layers(self, x):
        """
        Adds all deconvolution transpose blocks.

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
        Adds a deconvolution block consisting of a Conv2DTranspose, BN, LeakyReLU

        :param layer_index: Index of the deconvolution layer
        :param x: Previous layer
        :return: tf.keras.layer
        """
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index - 1],
            kernel_size=self.conv_kernels[layer_index - 1],
            strides=self.conv_strides[layer_index - 1],
            padding="same",
            kernel_regularizer=l2(0.001),
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        x = ReLU(name=f"decoder_leakyrelu_{layer_num}")(x)
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
            name=f"decoder_out_{self._num_conv_layers}"
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
        return Input(shape=self.input_shape, name="encoder_input"), \
               Input(shape=self.inf_vector_shape, name="encoder_inf_input")

    def _add_dense_to_inf(self, encoder_inf_input):
        """
        Adds Embedding and Dense layer to inf_vector input

        :param encoder_inf_input: Inf_vector model input
        :return: tf.keras.layer
        """
        n_neurons = self.n_neurons
        features = Embedding(2000, 256)(encoder_inf_input)
        x = Flatten()(features)
        x = Dense(n_neurons, name="encoder_inf_dense")(x)
        x = Dropout(.3)(x)
        return x

    def _add_conv_layers(self, encoder_input):
        """
        Adds all convolutional blocks

        :param encoder_input: Input layer
        :return: tf.keras.layer
        """
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        """
        Adds a convolutional block, consisting of Conv2D, BN, ReLU

        :param layer_index: Number of layer
        :param x: Previous layer
        :return: tf.keras.layer
        """
        layer_number = layer_index + 1
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            kernel_regularizer=l2(0.001),
            name=f"encoder_conv_layer_{layer_number}"
        )
        x = conv_layer(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        x = ReLU(name=f"encoder_relu_{layer_number}")(x)
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
        x = Dense(self.latent_space_dim, name="encoder_output")(x)
        return x


if __name__ == "__main__":
    autoencoder = Autoencoder(
        input_shape=(144, 160, 2),
        inf_vector_shape=(2, 16),
        conv_filters=(64, 128, 256, 512),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2),
        latent_space_dim=64,
        n_neurons=32 * 64,
        name="ae"
    )
    autoencoder.summary()
