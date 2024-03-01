import pandas as pd
from visualize import *
from dl_models.autoencoder import Autoencoder
from dl_models.vae import VAE
from dl_models.u_net import UNet
import numpy as np
from postprocess import PostProcess
from preprocess import Loader
import time
from tensorflow.keras import backend as K
import tensorflow as tf
from datageneratorv2 import DataGenerator
from dataset import Dataset

"""
Ignacio Martín 2022

Code corresponding to the Bachelor Thesis "Synthesis of Room Impulse Responses by Means of Deep Learning"

Authors:
    Ignacio Martín
    José Antonio Belloch
    Gema Piñero

University Carlos III de Madrid
"""

def amplitude_loss(y_true, y_pred):
    # loss = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred)) + 1.0e-12)
    loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    return loss

def phase_loss(y_true, y_pred):
    y_true = y_true * 2 * math.pi - math.pi
    y_pred = y_pred * 2 * math.pi - math.pi
    loss = tf.keras.backend.mean(1 - tf.math.cos(y_true - y_pred))
    return loss


if __name__ == '__main__':

    batch_size = 8
    debug = False
    target_size = (144, 160, 2)

    # ['ae', 'vae', 'resae', 'unet']

    models = ['ae_large_10']

    # models = ['ae_large_9', 'vae_large_9', 'resae_large_9', 'unet_large_9',
    #            'ae_large_9_sig', 'vae_large_9_sig', 'resae_large_9_sig', 'unet_large_9_sig']

    models_folder = '../results/mlsp/'
    saving_path = '../generated_rir_distributed/mlsp'
    rirs_path = '../../datasets/room_impulse/'
    loader = Loader(sample_rate=48000, mono=True, duration=0.2)

    rooms = ['ShoeBoxRoom']
    arrays = None
    algorithm = 'ph'
    diff_gen = False

    # Load data into RAM

    dataset = Dataset('../../datasets', 'room_impulse', normalization=True, debugging=debug, extract=False,
                      room_characteristics=True, room=rooms, array=arrays)
    test_generator = DataGenerator(dataset, batch_size=batch_size, partition='test', shuffle=False,
                                   characteristics=True)
    for model in models:
        name = model
        # Select model

        if 'vae' in model:
            print("Generating with VAE")
            trained_model = VAE(
                input_shape=target_size,
                inf_vector_shape=(2, 16),
                conv_filters=(64, 128, 256, 512),
                conv_kernels=(3, 3, 3, 3),
                conv_strides=(2, 2, 2, 2),
                latent_space_dim=32,
                n_neurons=32 * 64,
                name=name
            )

        elif 'ae' in model:
            print("Generating with AE")
            trained_model = Autoencoder(
                input_shape=target_size,
                inf_vector_shape=(2, 16),
                conv_filters=(64, 128, 256, 512),
                conv_kernels=(3, 3, 3, 3),
                conv_strides=(2, 2, 2, 2),
                latent_space_dim=64,
                n_neurons=32 * 64,
                name=name
            )

        elif 'unet' in model:
            print("Generating with UNET")
            trained_model = UNet(input_shape=target_size,
                                 inf_vector_shape=(2, 16),
                                 mode=0,
                                 number_filters_0=32,
                                 kernels=3,
                                 name=name
                                 )

        optimizer = tf.keras.optimizers.Adam()
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=trained_model.model)
        manager = tf.train.CheckpointManager(checkpoint, directory=models_folder + name, max_to_keep=1)
        checkpoint.restore(manager.latest_checkpoint)

        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        trained_model.summary()

        postprocessor = PostProcess(name, algorithm=algorithm)

        time.sleep(1)
        print('Generating wavs and obtaining loss')

        time_inference, time_postprocessing, time_loss = [], [], []
        total_loss, amp_loss, pha_loss, wav_loss, wav_loss_50ms, missa_amp_loss, missa_wav_loss = [], [], [], [], [], [], []

        hemi_total_loss, large_total_loss, medium_total_loss, shoe_total_loss, small_total_loss = [], [], [], [], []
        hemi_amp_loss, large_amp_loss, medium_amp_loss, shoe_amp_loss, small_amp_loss = [], [], [], [], []
        hemi_pha_loss, large_pha_loss, medium_pha_loss, shoe_pha_loss, small_pha_loss = [], [], [], [], []
        hemi_wav_loss, large_wav_loss, medium_wav_loss, shoe_wav_loss, small_wav_loss = [], [], [], [], []
        hemi_wav_loss_50ms, large_wav_loss_50ms, medium_wav_loss_50ms, shoe_wav_loss_50ms, small_wav_loss_50ms = [], [], [], [], []
        hemi_missa_amp_loss, large_missa_amp_loss, medium_missa_amp_loss, shoe_missa_amp_loss, small_missa_amp_loss = [], [], [], [], []
        hemi_missa_wav_loss, large_missa_wav_loss, medium_missa_wav_loss, shoe_missa_wav_loss, small_missa_wav_loss = [], [], [], [], []

        hemi_count, large_count, medium_count, shoe_count, small_count = 0, 0, 0, 0, 0

        plot_countdown = 0
        plot_count = 0

        start = time.time()

        for i in range(0, numUpdates):

            spec_in, emb, spec_out, characteristic = test_generator.__getitem__(i)

            start_inf = time.time()
            spec_generated = trained_model.model([spec_in, emb], training=False)
            end_inf = time.time()

            time_inference.append(end_inf - start_inf)

            for j in range(0, emb.shape[0]):
                start_gen = time.time()

                if diff_gen:
                    diff_phase_generated = (spec_generated[j, :, :, 1] + spec_in[j, :, :, 1]).numpy()
                    diff_spec_generated = np.stack((spec_generated[j, :, :, 0], diff_phase_generated), axis=-1)
                    wav_pred = postprocessor.post_process(diff_spec_generated, emb[j, 1, :])
                else:
                    wav_pred = postprocessor.post_process(spec_generated[j], emb[j, 1, :])

                end_gen = time.time()
                time_postprocessing.append(end_gen - start_gen)

                start_loss = time.time()

                stft_true = spec_out[j, :, :, 0]
                phase_true = spec_out[j, :, :, 1]

                stft_pred = spec_generated[j, :, :, 0]

                if diff_gen:
                    phase_pred = diff_phase_generated
                else:
                    phase_pred = spec_generated[j, :, :, 1]

                loss_stft = np.mean(amplitude_loss(stft_true, stft_pred))
                loss_phase = np.mean(phase_loss(phase_true, phase_pred))
                loss = np.mean(amplitude_loss(spec_out[j], spec_generated[j]))

                total_loss.append(loss)
                amp_loss.append(loss_stft)
                pha_loss.append(loss_phase)

                num = tf.norm((stft_pred - stft_true), ord=2)
                den = tf.norm(stft_true, ord=2)
                loss_missa_amp = 20 * math.log10(num / den)

                missa_amp_loss.append(loss_missa_amp)

                characteristic_out = characteristic[j, :, 1]
                wav_true = loader.load(
                    f'../../../datasets/room_impulse/{characteristic_out[0]}/Zone{characteristic_out[1]}/{characteristic_out[2]}'
                    f'MicrophoneArray/{characteristic_out[0]}_Zone{characteristic_out[1]}_'
                    f'{characteristic_out[2]}MicrophoneArray_L{characteristic_out[3]}_M{characteristic_out[4]}.wav')

                loss_wav = np.mean(amplitude_loss(wav_true, wav_pred))
                wav_loss.append(loss_wav)

                loss_wav_50ms = np.mean(amplitude_loss(wav_true[:2400], wav_pred[:2400]))
                wav_loss_50ms.append(loss_wav_50ms)

                num = tf.norm((wav_pred - wav_true), ord=2)
                den = tf.norm(wav_true, ord=2)
                loss_missa_wav = 20 * math.log10(num / den)

                missa_wav_loss.append(loss_missa_wav)

                if characteristic_out[0] == 'HemiAnechoicRoom':
                    hemi_count += 1

                    hemi_total_loss.append(loss)
                    hemi_amp_loss.append(loss_stft)
                    hemi_pha_loss.append(loss_phase)

                    hemi_wav_loss.append(loss_wav)
                    hemi_wav_loss_50ms.append(loss_wav_50ms)

                    hemi_missa_amp_loss.append(loss_missa_amp)
                    hemi_missa_wav_loss.append(loss_missa_wav)

                if characteristic_out[0] == 'LargeMeetingRoom':
                    large_count += 1

                    large_total_loss.append(loss)
                    large_amp_loss.append(loss_stft)
                    large_pha_loss.append(loss_phase)

                    large_wav_loss.append(loss_wav)
                    large_wav_loss_50ms.append(loss_wav_50ms)

                    large_missa_amp_loss.append(loss_missa_amp)
                    large_missa_wav_loss.append(loss_missa_wav)

                if characteristic_out[0] == 'MediumMeetingRoom':
                    medium_count += 1

                    medium_total_loss.append(loss)
                    medium_amp_loss.append(loss_stft)
                    medium_pha_loss.append(loss_phase)

                    medium_wav_loss.append(loss_wav)
                    medium_wav_loss_50ms.append(loss_wav_50ms)

                    medium_missa_amp_loss.append(loss_missa_amp)
                    medium_missa_wav_loss.append(loss_missa_wav)

                if characteristic_out[0] == 'ShoeBoxRoom':
                    shoe_count += 1

                    shoe_total_loss.append(loss)
                    shoe_amp_loss.append(loss_stft)
                    shoe_pha_loss.append(loss_phase)

                    shoe_wav_loss.append(loss_wav)
                    shoe_wav_loss_50ms.append(loss_wav_50ms)

                    shoe_missa_amp_loss.append(loss_missa_amp)
                    shoe_missa_wav_loss.append(loss_missa_wav)

                if characteristic_out[0] == 'SmallMeetingRoom':
                    small_count += 1

                    small_total_loss.append(loss)
                    small_amp_loss.append(loss_stft)
                    small_pha_loss.append(loss_phase)

                    small_wav_loss.append(loss_wav)
                    small_wav_loss_50ms.append(loss_wav_50ms)

                    small_missa_amp_loss.append(loss_missa_amp)
                    small_missa_wav_loss.append(loss_missa_wav)

                end_loss = time.time()
                time_loss.append(end_loss - start_loss)

                if plot_countdown == 640:
                    create_directory_if_none(f'../generated_rir_distributed/{name}_{algorithm}/png/')
                    plot_feature_vs_wav(stft_pred, wav_pred, name, characteristic_out,
                                        f'../generated_rir_distributed/{name}_{algorithm}/png/spec_vs_wav_{plot_count}.png')
                    plot_feature_vs_feature_wav(wav_true, stft_true, stft_pred, name, characteristic_out,
                                                f'../generated_rir_distributed/{name}_{algorithm}/png/spec_vs_spec_{plot_count}.png')
                    plot_phase_vs_phase(phase_true, phase_pred, name, characteristic_out,
                                        f'../generated_rir_distributed/{name}_{algorithm}/png/phase_vs_phase_{plot_count}.png')
                    plot_wav_vs_wav(wav_true, wav_pred, name, characteristic_out,
                                    f'../generated_rir_distributed/{name}_{algorithm}/png/wav_vs_wav_{plot_count}.png')
                    plot_count += 1
                    plot_countdown = 0
                else:
                    plot_countdown += 1

        end = time.time()
        total_loss = np.mean(total_loss)
        amp_loss = np.mean(amp_loss)
        pha_loss = np.mean(pha_loss)
        wav_loss = np.mean(wav_loss)
        wav_loss_50ms = np.mean(wav_loss_50ms)
        missa_amp_loss = np.mean(missa_amp_loss)
        missa_wav_loss = np.mean(missa_wav_loss)

        hemi_total_loss = np.mean(hemi_total_loss)
        hemi_amp_loss = np.mean(hemi_amp_loss)
        hemi_pha_loss = np.mean(hemi_pha_loss)
        hemi_wav_loss = np.mean(hemi_wav_loss)
        hemi_wav_loss_50ms = np.mean(hemi_wav_loss_50ms)
        hemi_missa_amp_loss = np.mean(hemi_missa_amp_loss)
        hemi_missa_wav_loss = np.mean(hemi_missa_wav_loss)

        large_total_loss = np.mean(large_total_loss)
        large_amp_loss = np.mean(large_amp_loss)
        large_pha_loss = np.mean(large_pha_loss)
        large_wav_loss = np.mean(large_wav_loss)
        large_wav_loss_50ms = np.mean(large_wav_loss_50ms)
        large_missa_amp_loss = np.mean(large_missa_amp_loss)
        large_missa_wav_loss = np.mean(large_missa_wav_loss)

        medium_total_loss = np.mean(medium_total_loss)
        medium_amp_loss = np.mean(medium_amp_loss)
        medium_pha_loss = np.mean(medium_pha_loss)
        medium_wav_loss = np.mean(medium_wav_loss)
        medium_wav_loss_50ms = np.mean(medium_wav_loss_50ms)
        medium_missa_amp_loss = np.mean(medium_missa_amp_loss)
        medium_missa_wav_loss = np.mean(medium_missa_wav_loss)

        shoe_total_loss = np.mean(shoe_total_loss)
        shoe_amp_loss = np.mean(shoe_amp_loss)
        shoe_pha_loss = np.mean(shoe_pha_loss)
        shoe_wav_loss = np.mean(shoe_wav_loss)
        shoe_wav_loss_50ms = np.mean(shoe_wav_loss_50ms)
        shoe_missa_amp_loss = np.mean(shoe_missa_amp_loss)
        shoe_missa_wav_loss = np.mean(shoe_missa_wav_loss)

        small_total_loss = np.mean(small_total_loss)
        small_amp_loss = np.mean(small_amp_loss)
        small_pha_loss = np.mean(small_pha_loss)
        small_wav_loss = np.mean(small_wav_loss)
        small_wav_loss_50ms = np.mean(small_wav_loss_50ms)
        small_missa_amp_loss = np.mean(small_missa_amp_loss)
        small_missa_wav_loss = np.mean(small_missa_wav_loss)

        time_inference = np.mean(time_inference[1:])
        time_postprocessing = np.mean(time_postprocessing[1:])
        time_loss = np.mean(time_loss[1:])

        time_data = {
            "n_samples": [numUpdates * emb.shape[0]],
            "t_model_inference_avg": [np.format_float_positional(time_inference, precision=5)],
            "batch_size": [emb.shape[0]],
            "t_postprocess": [np.format_float_positional(time_postprocessing, precision=5)],
            "t_loss_calc": [np.format_float_positional(time_loss, precision=5)],
            "t_global": [np.format_float_positional((end - start), precision=5)]
        }

        loss_data = {
            "room": ['Global', 'HemiAnechoic', 'Large', 'Medium', 'Shoe', 'Small'],
            "n samples": [numUpdates * emb.shape[0], hemi_count, large_count, medium_count, shoe_count, small_count],
            "MSE spectrogram": [np.format_float_positional(total_loss, precision=4),
                                np.format_float_positional(hemi_total_loss, precision=4),
                                np.format_float_positional(large_total_loss, precision=4),
                                np.format_float_positional(medium_total_loss, precision=4),
                                np.format_float_positional(shoe_total_loss, precision=4),
                                np.format_float_positional(small_total_loss, precision=4)],
            "MSE magnitude": [np.format_float_positional(amp_loss, precision=4),
                              np.format_float_positional(hemi_amp_loss, precision=4),
                              np.format_float_positional(large_amp_loss, precision=4),
                              np.format_float_positional(medium_amp_loss, precision=4),
                              np.format_float_positional(shoe_amp_loss, precision=4),
                              np.format_float_positional(small_amp_loss, precision=4)],
            "1-cos(y-y_) phase": [np.format_float_positional(pha_loss, precision=4),
                                  np.format_float_positional(hemi_pha_loss, precision=4),
                                  np.format_float_positional(large_pha_loss, precision=4),
                                  np.format_float_positional(medium_pha_loss, precision=4),
                                  np.format_float_positional(shoe_pha_loss, precision=4),
                                  np.format_float_positional(small_pha_loss, precision=4)],
            "MSE waveform": [np.format_float_scientific(wav_loss, precision=4),
                             np.format_float_scientific(hemi_wav_loss, precision=4),
                             np.format_float_scientific(large_wav_loss, precision=4),
                             np.format_float_scientific(medium_wav_loss, precision=4),
                             np.format_float_scientific(shoe_wav_loss, precision=4),
                             np.format_float_scientific(small_wav_loss, precision=4)],
            "MSE waveform 50ms": [np.format_float_scientific(wav_loss_50ms, precision=4),
                                  np.format_float_scientific(hemi_wav_loss_50ms, precision=4),
                                  np.format_float_scientific(large_wav_loss_50ms, precision=4),
                                  np.format_float_scientific(medium_wav_loss_50ms, precision=4),
                                  np.format_float_scientific(shoe_wav_loss_50ms, precision=4),
                                  np.format_float_scientific(small_wav_loss_50ms, precision=4)],
            "Misalignment magnitude": [np.format_float_scientific(missa_amp_loss, precision=4),
                                       np.format_float_scientific(hemi_missa_amp_loss, precision=4),
                                       np.format_float_scientific(large_missa_amp_loss, precision=4),
                                       np.format_float_scientific(medium_missa_amp_loss, precision=4),
                                       np.format_float_scientific(shoe_missa_amp_loss, precision=4),
                                       np.format_float_scientific(small_missa_amp_loss, precision=4)],
            "Misalignment waveform": [np.format_float_scientific(missa_wav_loss, precision=4),
                                      np.format_float_scientific(hemi_missa_wav_loss, precision=4),
                                      np.format_float_scientific(large_missa_wav_loss, precision=4),
                                      np.format_float_scientific(medium_missa_wav_loss, precision=4),
                                      np.format_float_scientific(shoe_missa_wav_loss, precision=4),
                                      np.format_float_scientific(small_missa_wav_loss, precision=4)]

        }

        time_dataframe = pd.DataFrame(time_data)
        loss_dataframe = pd.DataFrame(loss_data)


        time_dataframe.to_csv(f'../generated_rir_distributed/{name}_{algorithm}/{name}_infer_time.csv', index=False)
        loss_dataframe.to_csv(f'../generated_rir_distributed/{name}_{algorithm}/{name}_losses.csv', index=False)

        with open(f'../generated_rir_distributed/{name}_{algorithm}/{name}_results_inference.txt', 'w') as text_file:
            text_file.write(f'{name} results:\n\n')
            text_file.write(f'Took {np.format_float_positional(time_inference, precision=5)} '
                            f's on average to infer spectrograms with batch size of {emb.shape[0]}\n')
            text_file.write(f'Took {np.format_float_positional(time_postprocessing, precision=5)} '
                            f's on average to postprocess and generate each spectrogram and waveform\n')
            text_file.write(f'Took {np.format_float_positional(time_loss, precision=5)} '
                            f's on average to obtain the losses for each waveform\n')
            text_file.write(f'Took {np.format_float_positional((end - start), precision=5)} '
                            f's to generate, postprocess and obtain loss for '
                            f'{numUpdates * emb.shape[0]} samples\n')
            text_file.write('\n')

            text_file.write(f'Total losses:\n')

            text_file.write(f'Total loss: {np.format_float_positional(total_loss, precision=4)} (MSE whole spectrogram)'
                            f'\t|\tAmplitude loss: {np.format_float_positional(amp_loss, precision=4)} (MSE amplitude)'
                            f'\t|\tPhase loss: {np.format_float_positional(pha_loss, precision=4)} (1-cos(y_true - y_pred))'
                            f'\n')
            text_file.write(f'Waveform loss: {np.format_float_scientific(wav_loss, precision=4)} (MSE)'
                            f'\t|\t 50 ms waveform loss: {np.format_float_scientific(wav_loss_50ms, precision=4)} (MSE)'
                            f'\n')
            text_file.write(
                f'Misalignment loss (amplitude): {np.format_float_scientific(missa_amp_loss, precision=4)} (dB)'
                f'\t|\t Misalignment loss (wav): {np.format_float_scientific(missa_wav_loss, precision=4)} (dB)'
                f'\n')
            text_file.write('\n')

            text_file.write(f'HemiAnechoicRoom losses ({hemi_count} samples):\n')

            text_file.write(
                f'Total loss: {np.format_float_positional(hemi_total_loss, precision=4)} (MSE whole spectrogram)'
                f'\t|\tAmplitude loss: {np.format_float_positional(hemi_amp_loss, precision=4)} (MSE amplitude)'
                f'\t|\tPhase loss: {np.format_float_positional(hemi_pha_loss, precision=4)} (1-cos(y_true - y_pred))'
                f'\n')
            text_file.write(f'Waveform loss: {np.format_float_scientific(hemi_wav_loss, precision=4)} (MSE)'
                            f'\t|\t 50 ms waveform loss: {np.format_float_scientific(hemi_wav_loss_50ms, precision=4)} (MSE)'
                            f'\n')
            text_file.write(
                f'Misalignment loss (amplitude): {np.format_float_scientific(hemi_missa_amp_loss, precision=4)} (dB)'
                f'\t|\t Misalignment loss (wav): {np.format_float_scientific(hemi_missa_wav_loss, precision=4)} (dB)'
                f'\n')
            text_file.write('\n')

            text_file.write(f'LargeMeetingRoom losses ({large_count} samples):\n')

            text_file.write(
                f'Total loss: {np.format_float_positional(large_total_loss, precision=4)} (MSE whole spectrogram)'
                f'\t|\tAmplitude loss: {np.format_float_positional(large_amp_loss, precision=4)} (MSE amplitude)'
                f'\t|\tPhase loss: {np.format_float_positional(large_pha_loss, precision=4)} (1-cos(y_true - y_pred))'
                f'\n')
            text_file.write(f'Waveform loss: {np.format_float_scientific(large_wav_loss, precision=4)} (MSE)'
                            f'\t|\t 50 ms waveform loss: {np.format_float_scientific(large_wav_loss_50ms, precision=4)} (MSE)'
                            f'\n')
            text_file.write(
                f'Misalignment loss (amplitude): {np.format_float_scientific(large_missa_amp_loss, precision=4)} (dB)'
                f'\t|\t Misalignment loss (wav): {np.format_float_scientific(large_missa_wav_loss, precision=4)} (dB)'
                f'\n')
            text_file.write('\n')

            text_file.write(f'MediumMeetingRoom losses ({medium_count} samples):\n')

            text_file.write(
                f'Total loss: {np.format_float_positional(medium_total_loss, precision=4)} (MSE whole spectrogram)'
                f'\t|\tAmplitude loss: {np.format_float_positional(medium_amp_loss, precision=4)} (MSE amplitude)'
                f'\t|\tPhase loss: {np.format_float_positional(medium_pha_loss, precision=4)} (1-cos(y_true - y_pred))'
                f'\n')
            text_file.write(f'Waveform loss: {np.format_float_scientific(medium_wav_loss, precision=4)} (MSE)'
                            f'\t|\t 50 ms waveform loss: {np.format_float_scientific(medium_wav_loss_50ms, precision=4)} (MSE)'
                            f'\n')
            text_file.write(
                f'Misalignment loss (amplitude): {np.format_float_scientific(medium_missa_amp_loss, precision=4)} (dB)'
                f'\t|\t Misalignment loss (wav): {np.format_float_scientific(medium_missa_wav_loss, precision=4)} (dB)'
                f'\n')
            text_file.write('\n')

            text_file.write(f'ShoeBoxRoom losses ({shoe_count} samples):\n')

            text_file.write(
                f'Total loss: {np.format_float_positional(shoe_total_loss, precision=4)} (MSE whole spectrogram)'
                f'\t|\tAmplitude loss: {np.format_float_positional(shoe_amp_loss, precision=4)} (MSE amplitude)'
                f'\t|\tPhase loss: {np.format_float_positional(shoe_pha_loss, precision=4)} (1-cos(y_true - y_pred))'
                f'\n')
            text_file.write(f'Waveform loss: {np.format_float_scientific(shoe_wav_loss, precision=4)} (MSE)'
                            f'\t|\t 50 ms waveform loss: {np.format_float_scientific(shoe_wav_loss_50ms, precision=4)} (MSE)'
                            f'\n')
            text_file.write(
                f'Misalignment loss (amplitude): {np.format_float_scientific(shoe_missa_amp_loss, precision=4)} (dB)'
                f'\t|\t Misalignment loss (wav): {np.format_float_scientific(shoe_missa_wav_loss, precision=4)} (dB)'
                f'\n')
            text_file.write('\n')

            text_file.write(f'SmallMeetingRoom losses: ({small_count} samples)\n')

            text_file.write(
                f'Total loss: {np.format_float_positional(small_total_loss, precision=4)} (MSE whole spectrogram)'
                f'\t|\tAmplitude loss: {np.format_float_positional(small_amp_loss, precision=4)} (MSE amplitude)'
                f'\t|\tPhase loss: {np.format_float_positional(small_pha_loss, precision=4)} (1-cos(y_true - y_pred))'
                f'\n')
            text_file.write(f'Waveform loss: {np.format_float_scientific(small_wav_loss, precision=4)} (MSE)'
                            f'\t|\t 50 ms waveform loss: {np.format_float_scientific(small_wav_loss_50ms, precision=4)} (MSE) '
                            f'\n')
            text_file.write(
                f'Misalignment loss (amplitude): {np.format_float_scientific(small_missa_amp_loss, precision=4)} (dB)'
                f'\t|\t Misalignment loss (wav): {np.format_float_scientific(small_missa_wav_loss, precision=4)} (dB)'
                f'\n')

        print('Done! Clearing cache and allocated memory')
        del model
        K.clear_session()
