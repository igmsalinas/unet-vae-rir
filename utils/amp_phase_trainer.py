import json
import math
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


class Trainer:

    def __init__(self, alpha, n_epochs, optimizer, callbacks, lr_exp_decay, lr0, file_name):

        'Initialization'
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.lr0 = lr0
        self.model_checkpoint = callbacks[0]
        self.early_stop = callbacks[1]
        self.lr_exp_decay = lr_exp_decay[0]
        self.lr_exp_decay_epoch = lr_exp_decay[1]
        self.file_name = file_name

        'Secondary initialization'
        self.train_loss_history = np.empty((n_epochs, 3), dtype=np.float32)
        self.val_loss_history = np.empty((n_epochs, 3), dtype=np.float32)

        if 'nadam' in optimizer:
            self.optimizer = tf.keras.optimizers.Nadam(learning_rate=self.lr0)
        elif 'sgd' in optimizer:
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr0)
        elif 'adam' in optimizer:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr0)

    def train(self, model, train_generator, val_generator):
        # Train net
        print("[INFO]: Training model...")
        numUpdates = train_generator.__len__()
        numUpdates_val = val_generator.__len__()

        # Loop over all epochs
        for epoch in range(0, self.n_epochs):
            # Initialize variables
            train_loss, train_loss_phase, train_loss_stft = [], [], []
            val_loss, val_loss_phase, val_loss_stft = [], [], []

            # show the current epoch number
            print("\n[INFO]: Starting epoch {}/{}...".format(epoch + 1, self.n_epochs), end="\n")
            sys.stdout.flush()
            epochStart = time.time()
            # loop over the data in batch size increments

            # exponential lr in last epochs
            if self.lr_exp_decay:
                if epoch >= self.lr_exp_decay_epoch:
                    K.set_value(self.optimizer.learning_rate,
                                self.lr0 * np.exp(-0.25 * (epoch - self.lr_exp_decay_epoch)))

            # TRAINING
            for i in range(0, numUpdates):
                print("Training step: " + str(i) + "/" + str(numUpdates - 1), end='\r')
                # We take from our data generator one batch
                spec_in, spec_out, emb = train_generator.__next__()

                # Take a step in unet
                loss, loss_phase, loss_stft = self.step(spec_in, spec_out, emb, model)

                train_loss.append(loss)
                train_loss_phase.append(loss_phase)
                train_loss_stft.append(loss_stft)

            # VALIDATION
            for i in range(0, numUpdates_val):
                print("Validation step: " + str(i) + "/" + str(numUpdates_val - 1), end='\r')
                # We take from our data generator one batch
                spec_in, spec_out, emb = val_generator.__next__()

                spec_generated = model.model([spec_in, emb], training=False)
                loss, loss_phase, loss_stft = self.model_loss(spec_out, spec_generated)

                # Save loss
                val_loss.append(loss)
                val_loss_phase.append(loss_phase)
                val_loss_stft.append(loss_stft)

            # Show timing information for the epoch
            epochEnd = time.time()
            elapsed = (epochEnd - epochStart)
            print("took {:.4} seconds".format(elapsed))

            # Show training and validation loss for the epoch
            train_loss = np.mean(train_loss)
            train_loss_phase = np.mean(train_loss_phase)
            train_loss_stft = np.mean(train_loss_stft)
            print("Perdidas training:")
            print(" - Perdidas: " + str(train_loss))
            print(" - Perdidas fase: " + str(train_loss_phase))
            print(" - Perdidas módulo: " + str(train_loss_stft))

            val_loss = np.mean(val_loss)
            val_loss_phase = np.mean(val_loss_phase)
            val_loss_stft = np.mean(val_loss_stft)
            print("Perdidas validation:")
            print(" - Perdidas combinadas: " + str(val_loss))
            print(" - Perdidas fase: " + str(val_loss_phase))
            print(" - Perdidas módulo: " + str(val_loss_stft))

            # Save loss values
            self.train_loss_history[epoch][0] = train_loss
            self.train_loss_history[epoch][1] = train_loss_phase
            self.train_loss_history[epoch][2] = train_loss_stft
            self.val_loss_history[epoch][0] = val_loss
            self.val_loss_history[epoch][1] = val_loss_phase
            self.val_loss_history[epoch][2] = val_loss_stft

            # Callbacks execution
            improve = self.model_checkpoint.checkpoint(train_loss=train_loss, val_loss=val_loss, model=model)
            stop = self.early_stop.stop_count(improve=improve)

            if stop:
                break

        n_epochs = epoch + 1
        H = History(n_epochs, self.train_loss_history[:n_epochs, :], self.val_loss_history[:n_epochs, :])
        return model, H

    # Step function for gradient tape
    def step(self, spec_in, spec_out, emb, model):

        # keep track of our gradients
        with tf.GradientTape() as tape:
            spec_generated = model.model([spec_in, emb], training=True)
            loss, loss_phase, loss_stft = self.model_loss(spec_out, spec_generated)

        trainable_variables = model.model.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))

        return loss, loss_phase, loss_stft

    def model_loss(self, y_true, y_pred):
        # Generator predictions

        stft_true = y_true[:, :, :, 0]
        phase_true = y_true[:, :, :, 1]

        stft_pred = y_pred[:, :, :, 0]
        phase_pred = y_pred[:, :, :, 1]

        loss_stft = self.amplitude_loss(stft_true, stft_pred)
        loss_phase = self.phase_loss(phase_true, phase_pred)

        loss = loss_phase + loss_stft
        # loss = self.amplitude_loss(y_true, y_pred)
        return loss, loss_phase, loss_stft

    def amplitude_loss(self, y_true, y_pred):
        # loss = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred)) + 1.0e-12)
        loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred))
        return loss

    def phase_loss(self, y_true, y_pred):
        y_true = y_true * 2 * math.pi - math.pi
        y_pred = y_pred * 2 * math.pi - math.pi
        loss = tf.reduce_mean(tf.keras.backend.mean(1 - tf.math.cos(y_true - y_pred)))
        return loss


########################################################
# Callbacks
########################################################

class ModelCheckpoint(object):
    def __init__(self, filepath, save_best_only, verbose):
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.verbose = verbose

        'Secondary initialization'
        self.train_loss_min = 10
        self.val_loss_min = 10

    def checkpoint(self, train_loss, val_loss, model):
        improve = False
        if val_loss < self.val_loss_min:

            if self.verbose:
                print('Validation loss improved from ' + str(self.val_loss_min) + ' to ' + str(val_loss))

            if self.save_best_only:
                model.save(self.filepath)

            self.val_loss_min = val_loss
            self.train_loss_min = train_loss
            improve = True

        else:
            if self.verbose:
                print('Validation loss did not improve')

        return improve


class EarlyStopping(object):
    def __init__(self, patience):
        self.patience = patience

        'Secondary initialization'
        self.count = 0

    def stop_count(self, improve):
        stop = False
        if improve:
            self.count = 0
        else:
            self.count = self.count + 1

        if self.count == self.patience:
            stop = True

        return stop


class History(object):

    def __init__(self, epochs, train_loss_history, val_loss_history, train_acc_history=None, val_acc_history=None):
        'Save important parameters'
        self.epochs = epochs

        # In case of GAN training/validating:
        #   - Dimension [:][0] combined loss
        #   - Dimension [:][1] phase loss // reconstruction loss
        #   - Dimension [:][2] amplitude loss // kl_loss

        'Save training parameters'
        self.train_loss_history = train_loss_history
        self.train_acc_history = train_acc_history

        'Save validation parameters'
        self.val_loss_history = val_loss_history
        self.val_acc_history = val_acc_history


def plot_graphs(x1=None, y1=None, x2=None, y2=None, x3=None, y3=None, x4=None, y4=None, label1='', label2='', label3='',
                label4='', filename='./Graphic.png'):
    print('Representando grafica ...')
    if x1 is None:
        x1 = np.arange(0, len(y1))
    if y2 is not None and x2 is None:
        x2 = np.arange(0, len(y2))
    if y3 is not None and x3 is None:
        x3 = np.arange(0, len(y3))
    if y4 is not None and x4 is None:
        x4 = np.arange(0, len(y4))

    plt.style.use("ggplot")
    plt.figure()

    plt.plot(x1, y1, label=label1)
    if y2 is not None:
        plt.plot(x2, y2, label=label2)
    if y3 is not None:
        plt.plot(x3, y3, label=label3)
    if y4 is not None:
        plt.plot(x4, y4, label=label4)

    plt.title("Graphic")
    plt.xlabel("Epoch ")
    plt.ylabel("Loss")
    plt.legend()
    print('Guardando grafica')
    plt.savefig(filename)
    plt.close()


def params_saver(file_name, batch_size, optimizer, criterion, lr, BatchNorm, normalization, epochs,
                 callbacks, alpha, beta, number_filters_0):
    print('Creando diccionario hiperparametros')
    params = {}
    params['batch_size'] = batch_size
    params['optimizer'] = str(optimizer)
    params['criterion'] = str(criterion)
    params['epochs'] = epochs
    params['lr'] = lr
    params['alpha'] = alpha
    params['beta'] = beta
    params['batch_norm'] = BatchNorm
    params['normalization'] = normalization
    params['number_filters_0'] = number_filters_0
    params['val_loss'] = float(callbacks[0].val_loss_min)
    params['train_loss'] = float(callbacks[0].train_loss_min)
    params['patience'] = callbacks[1].patience

    with open(file_name + '/hiperparametros.json', 'w') as fp:
        json.dump(params, fp)


def rmse_coef(y_true, y_pred):
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)

    loss = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred)) + 1.0e-12)

    return loss
