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
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda, Embedding, concatenate, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

# import tensorflow_probability as tfp

class SamplingLayer(layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices

class VQVAE:
    """
    VAE represents a Deep Convolutional variational autoencoder architecture
    with mirrored encoder and decoder components with the addition of the information vector
    as an input
    """

    def __init__(self,
                 input_shape,
                 inf_vector_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim,
                 n_neurons,
                 name="VAE"
                 ):
        self.input_shape = input_shape  # Shape of the input tensor, the stacked spectrogram [128, 144, 2]
        self.inf_vector_shape = inf_vector_shape  # Shape of the information vector [10, 2]
        self.conv_filters = conv_filters  # Number of filters to be applied in each layer [32, 64, 128, 256]
        self.conv_kernels = conv_kernels  # Number of kernels per layer[3, 3, 3]
        self.conv_strides = conv_strides  # Number of strides used [2, 2, 2]
        self.latent_space_dim = latent_space_dim  # Dimension of the bottleneck 16
        self.n_neurons = n_neurons  # Number of neurons receiving the information vector
        self.name = name

        self.reconstruction_loss_weight = 100000  # Factor for reconstruction loss

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

        self.model.compile(optimizer=optimizer,
                           loss=self.calculate_combined_loss,
                           metrics=[self.calculate_reconstruction_loss,
                                    self.calculate_kl_loss])
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
        autoencoder = VAE(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder

    def calculate_combined_loss(self, y_target, y_predicted):
        """
        Obtains the total loss

        :param y_target: Target STFT
        :param y_predicted: Predicted STFT
        :return: Total Loss
        """
        reconstruction_loss = self.calculate_reconstruction_loss(y_target, y_predicted)
        kl_loss = self.calculate_kl_loss()
        combined_loss = reconstruction_loss + kl_loss
        return combined_loss, reconstruction_loss, kl_loss

    @staticmethod
    def calculate_reconstruction_loss(y_target, y_predicted):
        """
        Obtains the mean squared error

        :param y_target: Target STFT
        :param y_predicted: Predicted STFT
        :return: MSE Loss
        """
        error = y_target - y_predicted
        reconstruction_loss = K.mean(K.square(error), axis=(1, 2))
        return reconstruction_loss

    def calculate_kl_loss(self):
        """
        Obtains the KL divergence error

        :return: KL Loss
        """
        kl_loss = -0.5 * K.sum(1 + self.log_variance - K.square(self.mu) -
                               K.exp(self.log_variance), axis=1)
        return kl_loss

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
        self.model = Model(model_input, model_output, name="VAE")

    def _build_decoder(self):
        """
        Builds the Decoder
        """
        decoder_input = self._add_decoder_input()
        first_conv = self._add_first_conv(decoder_input)
        conv_transpose_layers = self._add_conv_transpose_layers(first_conv)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        """
        Adds the decoder input

        :return: tf.keras.layer
        """
        print(self.encoder.output.shape[1:])
        return Input(shape=self.encoder.output.shape[1:], name="decoder_input")


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
            name="decoder_conv_transpose_layer_0"
        )
        x = conv_transpose_layer(x)
        x = BatchNormalization(name=f"decoder_bn_0")(x)
        x = ReLU(name=f"decoder_relu_0")(x)

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
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        x = ReLU(name=f"decoder_relu_{layer_num}")(x)
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
        encoder_input, encoder_input_inf = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        dense_inf = self._add_dense_to_inf(encoder_input_inf)
        bottleneck = self._add_bottleneck(conv_layers, dense_inf)
        self._model_input = [encoder_input, encoder_input_inf]
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
        features = Embedding(1500, 128)(encoder_inf_input)
        x = Dense(n_neurons, name="encoder_inf_dense")(features)
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
            name=f"encoder_conv_layer_{layer_number}"
        )
        x = conv_layer(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        x = ReLU(name=f"encoder_relu_{layer_number}")(x)
        return x

    def _add_bottleneck(self, x, y):
        """
        Flattens the data, concatenates it and applies Gaussian distribution
        by means of Sampling layer "function(latent space + latent space)"

        :param x: Last convolutional block output
        :param y: Dense output from inf_vector
        :return: tf.keras.layer
        """
        self._shape_before_bottleneck = K.int_shape(x)[1:]

        shape = x.get_shape().as_list()[1:]
        shape[2] = 2

        x = Flatten()(x)
        y = Flatten()(y)
        latent_space = concatenate([x, y])

        dim = np.prod(shape)

        x = Dense(dim)(latent_space)
        x = Dropout(.3)(x)
        x = Reshape(shape)(x)

        x = Conv2D(self.conv_filters[-1], (1, 1))(x)

        vq_layer = VectorQuantizer(self.conv_filters[-1], self.latent_space_dim, name="vector_quantizer")

        x = vq_layer(x)

        return x
if __name__ == "__main__":
    tiny_vae = VQVAE(
        input_shape=(160, 144, 2),
        inf_vector_shape=(16, 2),
        conv_filters=(32, 64, 128, 256),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2),
        latent_space_dim=16,
        n_neurons=320,
        name="VAE"
    )

    tiny_vae.summary()
