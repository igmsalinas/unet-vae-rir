import json
import math
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from distributed_scripts.utils.trainer import History


class VAETrainer:

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
            train_loss, train_loss_rec, train_loss_kl = [], [], []
            val_loss, val_loss_rec, val_loss_kl = [], [], []

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
                comb_loss, reconstruction_loss, kl_loss = self.step(spec_in, spec_out, emb, model)

                train_loss.append(comb_loss)
                train_loss_rec.append(reconstruction_loss)
                train_loss_kl.append(kl_loss)

            # VALIDATION
            for i in range(0, numUpdates_val):
                print("Validation step: " + str(i) + "/" + str(numUpdates_val - 1), end='\r')
                # We take from our data generator one batch
                spec_in, spec_out, emb = val_generator.__next__()

                z, mean, log_var = model.encoder([spec_in, emb], training=False)
                spec_generated = model.decoder(z)
                reconstruction_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(spec_out, spec_generated))
                kl_loss = -0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                comb_loss = reconstruction_loss + kl_loss

                # Save loss
                val_loss.append(comb_loss)
                val_loss_rec.append(reconstruction_loss)
                val_loss_kl.append(kl_loss)

            # Show timing information for the epoch
            epochEnd = time.time()
            elapsed = (epochEnd - epochStart)
            print("took {:.4} seconds".format(elapsed))

            # Show training and validation loss for the epoch
            train_loss = np.mean(train_loss)
            train_loss_rec = np.mean(train_loss_rec)
            train_loss_kl = np.mean(train_loss_kl)
            print("Training losses:")
            print(" - Combined losses: " + str(train_loss))
            print(" - Reconstruction losses: " + str(train_loss_rec))
            print(" - KL losses: " + str(train_loss_kl))

            val_loss = np.mean(val_loss)
            val_loss_rec = np.mean(val_loss_rec)
            val_loss_kl = np.mean(val_loss_kl)
            print("Validation losses:")
            print(" - Combined losses: " + str(val_loss))
            print(" - Reconstruction losses: " + str(val_loss_rec))
            print(" - KL losses: " + str(val_loss_kl))

            # Save loss values
            self.train_loss_history[epoch][0] = train_loss
            self.train_loss_history[epoch][1] = train_loss_rec
            self.train_loss_history[epoch][2] = train_loss_kl
            self.val_loss_history[epoch][0] = val_loss
            self.val_loss_history[epoch][1] = val_loss_rec
            self.val_loss_history[epoch][2] = val_loss_kl

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
            z, mean, log_var = model.encoder([spec_in, emb], training=True)
            spec_generated = model.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(spec_out, spec_generated))
            kl_loss = -0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            comb_loss = reconstruction_loss + kl_loss

        trainable_variables = model.model.trainable_variables
        grads = tape.gradient(comb_loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))

        return comb_loss, reconstruction_loss, kl_loss

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
