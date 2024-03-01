import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy

class deep_CNN:

    def __init__(self, width, height, depth, classes, batchNorm, optimizer, learning_rate):

        self.width = width
        self.height = height
        self.depth = depth
        self.classes = classes
        self.batchNorm = batchNorm
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        self.model = None
        self.criterion = None

    def build_model(self):

        # First Input
        inputs = tf.keras.layers.Input(shape=(self.width, self.height, self.depth))

        # Layers convolutional, batchNorm, pooling
        x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', name='Conv0_A')(inputs)
        if self.batchNorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)

        # Layers convolutional, batchNorm, pooling
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', name='Conv1_A')(x)
        if self.batchNorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)

        # Layers convolutional, batchNorm, pooling
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', name='Conv0_C')(x)
        if self.batchNorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        # Capa FC => Relu
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        if self.batchNorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(rate=0.5)(x)

        # Clasificador softmax
        predictions = tf.keras.layers.Dense(self.classes, activation='softmax')(x)

        # Establezco el modelo y lo devuelvo
        model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
        self.model = model

    def set_optimizer_criterion(self):
        if 'nadam' in self.optimizer:
            optimizer = tf.keras.optimizers.Nadam(lr=self.learning_rate)
        elif 'sgd' in self.optimizer:
            optimizer = tf.keras.optimizers.SGD(lr=self.learning_rate)
        elif 'adam' in self.optimizer:
            optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)

        self.criterion = categorical_crossentropy

    def return_params(self):
        return self.model, self.optimizer, self.criterion