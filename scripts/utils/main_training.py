import math
import os
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from datageneratorv2 import DataGenerator
from dataset import Dataset
from dl_models.u_net import UNet
from dl_models.u_net_new import UNetN
from dl_models.autoencoder import Autoencoder
from dl_models.res_ae import ResAE
from dl_models.vae import VAE
from dl_models.unet_vae import UNetVAE
from dl_models.unet_vae_emb import UNetVAEEmb
import time
from tensorflow.keras.utils import Progbar


def sigmoid(beta, dimensions):
    x = np.linspace(-10, 10, dimensions[1])
    z = 1 / (1 + np.exp(-(x + 5) * beta))
    z = np.flip(z)
    sig = np.tile(z, (dimensions[0], 1))
    return sig


if __name__ == '__main__':

    ########################################################
    # Inputs and model selection
    ########################################################

    debug = False

    target_size = (144, 160, 2)
    rooms = None
    arrays = ["PlanarMicrophoneArray"]
    zones = None

    name = 'unet-vae-emb'

    diff = True

    if diff:
        diff_str = "-diff"
        diff_loss = True
    else:
        diff_str = ""
        diff_loss = False

    if name in ["unet-vae", "unet-n"]:
        normalize_vector = True
    else:
        normalize_vector = False

    loss = "mse"  # MAE or MSE
    alpha = 0.9

    mode = 3
    latent_space_dim = 64

    modifier = f"-{latent_space_dim}-{loss}{diff_str}"

    ########################################################
    # Hyperparams -  MODEL TRAINING
    ########################################################

    sigmoid_loss = False
    beta = 0.5

    n_epochs = 100
    lr = 5e-7
    batch_size_per_replica = 16

    optimizer_sel = "adam"
    lr_exp_decay = [True, 80]

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

    sig_size = (target_size[0], target_size[1])
    sig = sigmoid(beta, sig_size)
    sig = np.repeat(np.expand_dims(sig, 0), batch_size_per_replica, axis=0)

    strategy = tf.distribute.MirroredStrategy()

    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    global_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

    ########################################################
    # Data directories and folders
    ########################################################

    file_name = '../results/' + name + modifier
    if not os.path.exists(file_name):
        os.makedirs(file_name)

    ########################################################
    # Main training
    ########################################################

    if normalize_vector:
        dtype = tf.float32
    else:
        dtype = tf.int32

    # Prepare data generators
    dataset = Dataset('../../../datasets', 'room_impulse', normalization=True, debugging=debug,
                      extract=False, room=rooms, array=arrays, zone=zones, normalize_vector=normalize_vector)

    train_generator = DataGenerator(dataset, batch_size=global_batch_size, partition='train', shuffle=True,
                                    normalize_vector=normalize_vector)
    val_generator = DataGenerator(dataset, batch_size=global_batch_size, partition='val', shuffle=False,
                                  normalize_vector=normalize_vector)


    def generator_t(stop):
        i = 0
        while i < stop:
            spec_in, emb, spec_out = train_generator.__getitem__(i)
            yield spec_in, emb, spec_out
            i += 1


    def generator_v(stop):
        i = 0
        while i < stop:
            spec_in, emb, spec_out = val_generator.__getitem__(i)
            yield spec_in, emb, spec_out
            i += 1


    train_dataset = tf.data.Dataset.from_generator(generator_t, args=[train_generator.__len__()],
                                                   output_types=(tf.float32, dtype, tf.float32),
                                                   output_shapes=(tf.TensorShape(
                                                       [None, target_size[0], target_size[1], target_size[2]]),
                                                                  tf.TensorShape([None, 2, 16]),
                                                                  tf.TensorShape([None, target_size[0], target_size[1],
                                                                                  target_size[2]])))

    val_dataset = tf.data.Dataset.from_generator(generator_v, args=[val_generator.__len__()],
                                                 output_types=(tf.float32, dtype, tf.float32),
                                                 output_shapes=(tf.TensorShape(
                                                     [None, target_size[0], target_size[1], target_size[2]]),
                                                                tf.TensorShape([None, 2, 16]),
                                                                tf.TensorShape([None, target_size[0], target_size[1],
                                                                                target_size[2]])))

    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)

    with strategy.scope():

        if name == "ae":
            # Create model
            model = Autoencoder(
                input_shape=target_size,
                inf_vector_shape=(2, 16),
                conv_filters=(64, 128, 256, 512),
                conv_kernels=(3, 3, 3, 3),
                conv_strides=(2, 2, 2, 2),
                latent_space_dim=64,
                n_neurons=32 * 64,
                name=name
            )
        elif name == "resae":
            model = ResAE(
                input_shape=target_size,
                inf_vector_shape=(2, 16),
                conv_filters=(32, 64, 128, 256),
                conv_kernels=(3, 3, 3, 3),
                conv_strides=(2, 2, 2, 2),
                latent_space_dim=32,
                n_neurons=16 * 64,
                name=name
            )
        elif name == "vae":
            model = VAE(
                input_shape=target_size,
                inf_vector_shape=(2, 16),
                conv_filters=(64, 128, 256, 512),
                conv_kernels=(3, 3, 3, 3),
                conv_strides=(2, 2, 2, 2),
                latent_space_dim=64,
                n_neurons=32 * 64,
                name=name
            )

        elif name == "unet":
            model = UNet(input_shape=target_size,
                         inf_vector_shape=(2, 16),
                         mode=3,
                         number_filters_0=32,
                         kernels=3,
                         name="UNet"
                         )

        elif name == "unet-n":
            model = UNetN(input_shape=target_size,
                          inf_vector_shape=(2, 16),
                          mode=mode,
                          number_filters_0=32,
                          kernels=3,
                          latent_space_dim=latent_space_dim,
                          name="UNetN"
                          )

        elif name == "unet-vae":
            model = UNetVAE(input_shape=(144, 160, 2),
                            inf_vector_shape=(2, 16),
                            mode=mode,
                            number_filters_0=32,
                            kernels=3,
                            latent_space_dim=latent_space_dim/2,
                            name='UnetVAE'
                            )

        elif name == "unet-vae-emb":
            model = UNetVAEEmb(input_shape=(144, 160, 2),
                            inf_vector_shape=(2, 16),
                            mode=mode,
                            number_filters_0=32,
                            kernels=3,
                            latent_space_dim=latent_space_dim/2,
                            name='UnetVAEEmb'
                            )

        # Set optimizer and checkpoint
        if 'nadam' in optimizer_sel:
            optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
        elif 'sgd' in optimizer_sel:
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        elif 'adam' in optimizer_sel:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model.model)
        manager = tf.train.CheckpointManager(checkpoint, directory=file_name, max_to_keep=1)

        model.summary()

        # Define the loss function

        if loss == "mse":
            loss_object_amplitude = tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.NONE)
        elif loss == "mae":
            loss_object_amplitude = tf.keras.losses.MeanAbsoluteError(
                reduction=tf.keras.losses.Reduction.NONE)

        def phase_loss(y_true, y_pred):
            y_true = y_true * 2 * math.pi - math.pi
            y_pred = y_pred * 2 * math.pi - math.pi
            y_diff = y_true - y_pred
            phase = (y_diff + math.pi) % (2 * math.pi) - math.pi
            loss = 1 - tf.math.cos(phase)
            return loss


        def kl_loss_object(mean, log_var):
            kl_loss = -0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))
            return kl_loss


        def compute_kl_loss(mean, log_var):
            kl_loss = kl_loss_object(mean, log_var)
            per_example_loss = tf.reduce_sum(kl_loss, axis=1)
            kl_loss = tf.nn.compute_average_loss(per_example_loss,
                                                 global_batch_size=global_batch_size)
            return kl_loss


        def compute_loss(x, y_true, y_pred, model_losses):
            stft_true = y_true[:, :, :, 0]
            phase_true = y_true[:, :, :, 1]
            stft_pred = y_pred[:, :, :, 0]
            phase_pred = y_pred[:, :, :, 1]

            phase_x = x[:, :, :, 1]

            per_example_loss_amplitude = loss_object_amplitude(tf.expand_dims(stft_true, -1),
                                                               tf.expand_dims(stft_pred, -1))

            if diff_loss:

                per_example_loss_phase = phase_loss(phase_true - phase_x, phase_pred)
            else:
                per_example_loss_phase = phase_loss(phase_true, phase_pred)


            if sigmoid_loss:
                per_example_loss_phase = per_example_loss_phase * tf.convert_to_tensor(sig, np.float32)

            per_example_loss = alpha * per_example_loss_amplitude + (1 - alpha) * per_example_loss_phase

            per_example_loss /= tf.cast(tf.reduce_prod(tf.shape(y_true)[1:]), tf.float32)

            loss = tf.nn.compute_average_loss(per_example_loss,
                                              global_batch_size=global_batch_size)
            if model_losses:
                loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))

            return loss


        # Define metrics
        train_loss_amplitude = tf.keras.metrics.Mean(
            name='train_loss_amplitude')
        train_loss_phase = tf.keras.metrics.Mean(name='train_loss_phase')

        val_loss_amplitude = tf.keras.metrics.Mean(name='val_loss_amplitude')
        val_loss_phase = tf.keras.metrics.Mean(name='val_loss_phase')

        # VAE metrics

        if name == "vae" or name == "unet-vae":
            train_loss_kl = tf.keras.metrics.Mean(name='train_loss_kl')
            val_loss_kl = tf.keras.metrics.Mean(name='val_loss_kl')


    def train_step(inputs):
        spec_in, emb, spec_out = inputs

        with tf.GradientTape() as tape:
            if name == "vae":
                z, mean, log_var = model.encoder([spec_in, emb], training=True)
                spec_pred = model.decoder(z, training=True)
            elif name == "unet-vae":
                spec_pred, mean, log_var = model.model([spec_in, emb], training=True)
            else:
                spec_pred = model.model([spec_in, emb], training=True)

            loss = compute_loss(spec_in, spec_out, spec_pred, model.model.losses)

            if name == "vae" or name == "unet-vae":
                loss += compute_kl_loss(mean, log_var)

        gradients = tape.gradient(loss, model.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.model.trainable_variables))

        stft_true = spec_out[:, :, :, 0]
        phase_true = spec_out[:, :, :, 1]
        stft_pred = spec_pred[:, :, :, 0]
        phase_pred = spec_pred[:, :, :, 1]

        loss_amplitude = loss_object_amplitude(tf.expand_dims(stft_true, -1),
                                               tf.expand_dims(stft_pred, -1))

        if diff_loss:
            loss_phase = phase_loss(phase_true - spec_in[:, :, :, 1], phase_pred)
        else:
            loss_phase = phase_loss(phase_true, phase_pred)

        train_loss_amplitude.update_state(loss_amplitude)
        train_loss_phase.update_state(loss_phase)

        if name == "vae" or name == "unet-vae":
            loss_kl = kl_loss_object(mean, log_var)
            train_loss_kl.update_state(loss_kl)

        return loss


    def test_step(inputs):
        spec_in, emb, spec_out = inputs

        if name == "vae":
            z, mean, log_var = model.encoder([spec_in, emb], training=True)
            spec_pred = model.decoder(z, training=True)
        elif name == "unet-vae":
            spec_pred, mean, log_var = model.model([spec_in, emb], training=True)
        else:
            spec_pred = model.model([spec_in, emb], training=True)

        stft_true = spec_out[:, :, :, 0]
        phase_true = spec_out[:, :, :, 1]
        stft_pred = spec_pred[:, :, :, 0]
        phase_pred = spec_pred[:, :, :, 1]

        loss_amplitude = loss_object_amplitude(tf.expand_dims(stft_true, -1),
                                               tf.expand_dims(stft_pred, -1))

        if diff_loss:
            loss_phase = phase_loss(phase_true - spec_in[:, :, :, 1], phase_pred)
        else:
            loss_phase = phase_loss(phase_true, phase_pred)

        loss = compute_loss(spec_in, spec_out, spec_pred, model.model.losses)

        val_loss_amplitude.update_state(loss_amplitude)
        val_loss_phase.update_state(loss_phase)

        if name == "vae" or name == "unet-vae":
            loss_kl = kl_loss_object(mean, log_var)
            val_loss_kl.update_state(loss_kl)
            loss += compute_kl_loss(mean, log_var)

        return loss
    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)


    @tf.function
    def distributed_test_step(dataset_inputs):
        per_replica_losses = strategy.run(test_step, args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)


    start = time.time()

    patience = 10
    wait = 0
    best = float('inf')

    for epoch in range(n_epochs):
        # TRAIN LOOP

        print("\nEpoch {}/{}".format(epoch + 1, n_epochs))

        progBar = Progbar(train_generator.__len__())

        epoch_start = time.time()

        # exponential lr in last epochs
        if lr_exp_decay[0]:
            if epoch >= lr_exp_decay[1]:
                K.set_value(optimizer.learning_rate, lr * 0.9 ** (epoch / lr_exp_decay[1]))

        total_loss = 0.0
        num_batches = 0

        for x in train_dist_dataset:
            total_loss += distributed_train_step(x)
            num_batches += 1

            progBar.update(num_batches)

        train_loss = total_loss / num_batches

        # TEST LOOP
        total_val_loss = 0.0
        val_batches = 0

        for x in val_dist_dataset:
            total_val_loss += distributed_test_step(x)
            val_batches += 1

        val_loss = total_val_loss / num_batches

        progBar.update(train_generator.__len__(), finalize=True)

        if epoch % 2 == 0:
            save_path = manager.save()

        epoch_end = time.time()

        if name == "vae" or name == "unet-vae":
            template = ("Epoch {}, Loss: {}, Epoch time: {}\n"
                        "Train | {} Loss: {}, Phase Loss: {}, KL Loss: {}\n"
                        "Val   | {} Loss: {}, Phase Loss: {}, KL Loss: {}\n"
                        "lr    | {}")
            print(template.format(epoch + 1, train_loss, (epoch_end - epoch_start),
                                  loss, train_loss_amplitude.result(), train_loss_phase.result(), train_loss_kl.result(),
                                  loss, val_loss_amplitude.result(), val_loss_phase.result(), val_loss_kl.result(),
                                  optimizer.lr.numpy()))
        else:
            template = ("Epoch {}, Loss: {}, Epoch time: {}\n"
                        "Train | {} Loss: {}, Phase Loss: {}\n"
                        "Val   | {} Loss: {}, Phase Loss: {}\n"
                        "lr    | {}")
            print(template.format(epoch + 1, train_loss, (epoch_end - epoch_start),
                                  loss, train_loss_amplitude.result(), train_loss_phase.result(),
                                  loss, val_loss_amplitude.result(), val_loss_phase.result(),
                                  optimizer.lr.numpy()))

        train_loss_amplitude.reset_states()
        train_loss_phase.reset_states()
        val_loss_amplitude.reset_states()
        val_loss_phase.reset_states()

        wait += 1
        if val_loss < best:
            print(f"Improved loss = {val_loss}")
            best = val_loss
            wait = 0
        if wait >= patience:
            break

    end = time.time()
    print("Training complete, took " + str(end - start))

