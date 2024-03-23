import math
import pandas as pd
from visualize import *
from datageneratorv2 import DataGenerator
from dataset import Dataset
from dl_models.autoencoder import Autoencoder
from dl_models.vae import VAE
from dl_models.res_ae import ResAE
from dl_models.u_net import UNet
from dl_models.unet_vae import UNetVAE
from dl_models.u_net_new import UNetN
from dl_models.unet_vae_emb import UNetVAEEmb
import numpy as np
from numpy.fft import fft, ifft
from postprocess import PostProcess
from preprocess import Loader
import time
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.utils import Progbar

"""
Ignacio Martín 2024

Authors:
    Ignacio Martín
    José Antonio Belloch
    Gema Piñero

University Carlos III de Madrid
"""


def amplitude_loss(y_true, y_pred):
    return tf.keras.losses.mse(y_true, y_pred)


def phase_loss(y_true, y_pred):
    y_true_adj = y_true * 2 * math.pi - math.pi
    y_pred_adj = y_pred * 2 * math.pi - math.pi
    return tf.keras.backend.mean(1 - tf.math.cos(y_true_adj - y_pred_adj))


def sdr(pred, true):
    pred_adj = pred - np.mean(pred)
    true_adj = true - np.mean(true)
    origin_power = np.sum(true_adj ** 2) + 1e-8
    scale = np.sum(true_adj * pred_adj) / origin_power
    est_true = scale * true_adj
    est_res = pred_adj - est_true
    true_power = np.sum(est_true ** 2)
    res_power = np.sum(est_res ** 2)
    return 10 * np.log10(true_power) - 10 * np.log10(res_power)


def calculate_similarity(signal_a, signal_b, weights=None):
    """
    Compares two signals using various similarity measures and combines them into a single metric.

    Parameters:
    - signal_a: np.array, the first signal.
    - signal_b: np.array, the second signal.
    - weights: dict, optional; weights for each similarity measure.

    Returns:
    - metric: float, a composite metric evaluating the similarity between the two signals.
    """
    if weights is None:
        weights = {
            'time_static': 1.0,
            'time_shift': 1.0,
            'freq_static': 1.0,
            'freq_shift': 1.0,
            'energy': 1.0,
        }

    def time_domain_similarity_static():
        return np.sum(signal_a * signal_b)

    def time_domain_similarity_shift():
        fft_a = fft(signal_a)
        fft_b = fft(signal_b)
        product = fft_a * np.conj(fft_b)
        return np.sum(np.abs(ifft(product)))

    def frequency_domain_similarity_static():
        fft_a = fft(signal_a)
        fft_b = fft(signal_b)
        return np.sum(np.abs(fft_a * np.conj(fft_b)))

    def frequency_domain_similarity_shift():
        product = signal_a * signal_b
        fft_product = fft(product)
        return np.sum(np.abs(fft_product))

    def energy_similarity():
        power_a = np.sum(np.square(signal_a)) / len(signal_a)
        power_b = np.sum(np.square(signal_b)) / len(signal_b)
        return np.abs(power_a - power_b)

    # Compute similarities
    similarities = {
        'time_static': time_domain_similarity_static(),
        'time_shift': time_domain_similarity_shift(),
        'freq_static': frequency_domain_similarity_static(),
        'freq_shift': frequency_domain_similarity_shift(),
        'energy': energy_similarity(),
    }

    # Normalize and weight similarities
    max_vals = {key: max(1, np.abs(val)) for key, val in similarities.items()}  # Avoid division by zero
    weighted_sum = sum(weights[key] * (similarities[key] / max_vals[key]) for key in similarities)
    total_weight = sum(weights.values())

    # Compute final metric
    metric = weighted_sum / total_weight

    return metric


if __name__ == '__main__':

    # ['ae', 'vae', 'resae', 'unet', 'unet-n', 'unet-vae']
    model_name = "unet-vae-emb"
    latent_space_dim = 128
    loss = "mae"
    diff = True

    batch_size = 16
    debug = False

    rooms = None
    arrays = ["PlanarMicrophoneArray"]
    zones = None

    algorithms = ['ph', 'gl_ph', 'gl_mag']  # ['gl_ph', 'gl_mag', 'ph']
    n_iters = 64
    momentum = 0.99

    target_size = (144, 160, 2)
    mode = 3

    if diff:
        diff_str = "-diff"
        diff_gen = True
    else:
        diff_str = ""
        diff_gen = False

    modifier = f"-{latent_space_dim}-{loss}{diff_str}"

    if model_name in ["unet-vae", "unet-n"]:
        normalize_vector = True
    else:
        normalize_vector = False

    dataset_dir = '../../../datasets'
    models_folder = '../results/'
    saving_path = '../generated_rir/' + model_name + modifier

    if 'unet-n' == model_name:
        print("Generating with UNET-N")
        trained_model = UNetN(input_shape=target_size,
                              inf_vector_shape=(2, 16),
                              mode=mode,
                              number_filters_0=32,
                              kernels=3,
                              latent_space_dim=latent_space_dim,
                              name=model_name + modifier
                              )

    elif 'unet-vae' == model_name:
        print("Generating with UNET-VAE")
        trained_model = UNetVAE(input_shape=target_size,
                                inf_vector_shape=(2, 16),
                                mode=mode,
                                number_filters_0=32,
                                kernels=3,
                                latent_space_dim=latent_space_dim / 2,
                                name=model_name + modifier
                                )

    elif 'unet-vae-emb' == model_name:
        trained_model = UNetVAEEmb(input_shape=target_size,
                           inf_vector_shape=(2, 16),
                           mode=mode,
                           number_filters_0=32,
                           kernels=3,
                           latent_space_dim=latent_space_dim / 2,
                           name=model_name + modifier
                           )

    elif 'vae' == model_name:
        print("Generating with VAE")
        trained_model = VAE(
            input_shape=target_size,
            inf_vector_shape=(2, 16),
            conv_filters=(64, 128, 256, 512),
            conv_kernels=(3, 3, 3, 3),
            conv_strides=(2, 2, 2, 2),
            latent_space_dim=32,
            n_neurons=32 * 64,
            name=model_name
        )

    elif 'resae' == model_name:
        print("Generating with RESAE")
        trained_model = ResAE(
            input_shape=target_size,
            inf_vector_shape=(2, 16),
            conv_filters=(32, 64, 128, 256),
            conv_kernels=(3, 3, 3, 3),
            conv_strides=(2, 2, 2, 2),
            latent_space_dim=64,
            n_neurons=32 * 64,
            name=model_name
        )

    elif 'ae' == model_name:
        print("Generating with AE")
        trained_model = Autoencoder(
            input_shape=target_size,
            inf_vector_shape=(2, 16),
            conv_filters=(64, 128, 256, 512),
            conv_kernels=(3, 3, 3, 3),
            conv_strides=(2, 2, 2, 2),
            latent_space_dim=64,
            n_neurons=32 * 64,
            name=model_name
        )

    elif 'unet' == model_name:
        print("Generating with UNET")
        trained_model = UNet(input_shape=target_size,
                             inf_vector_shape=(2, 16),
                             mode=3,
                             number_filters_0=32,
                             kernels=3,
                             name=model_name
                             )

    physical_devices = tf.config.list_physical_devices('GPU')
    print(f"\n Device(s) : {tf.config.experimental.get_device_details(physical_devices[0])['device_name']} \n")

    optimizer = tf.keras.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=trained_model.model)
    manager = tf.train.CheckpointManager(checkpoint, directory=models_folder + model_name + modifier, max_to_keep=1)
    checkpoint.restore(manager.latest_checkpoint)

    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    trained_model.summary()

    loader = Loader(sample_rate=48000, mono=True, duration=0.2)

    # Load data into RAM

    dataset = Dataset(dataset_dir, 'room_impulse', normalization=True, debugging=debug, extract=False,
                      room_characteristics=True, room=rooms, array=arrays, zone=zones,
                      normalize_vector=normalize_vector)
    test_generator = DataGenerator(dataset, batch_size=batch_size, partition='test', shuffle=False,
                                   characteristics=True)
    if normalize_vector:
        min_max_vector = dataset.return_min_max_vector()
    else:
        min_max_vector = None

    for algorithm in algorithms:

        postprocessor = PostProcess(folder=saving_path + "/" + model_name + modifier, algorithm=algorithm,
                                    momentum=momentum, n_iters=n_iters, normalize_vector=normalize_vector)

        time.sleep(1)
        print(f'Generating wavs and obtaining loss | {algorithm}')
        numUpdates = test_generator.__len__()
        time_inference, time_postprocessing, time_loss = [], [], []
        (total_loss, amp_loss, pha_loss, wav_loss, wav_loss_50ms,
         missa_amp_loss, missa_wav_loss, sdr_metric, similarity_metric) = [], [], [], [], [], [], [], [], []

        hemi_total_loss, large_total_loss, medium_total_loss, shoe_total_loss, small_total_loss = [], [], [], [], []
        hemi_amp_loss, large_amp_loss, medium_amp_loss, shoe_amp_loss, small_amp_loss = [], [], [], [], []
        hemi_pha_loss, large_pha_loss, medium_pha_loss, shoe_pha_loss, small_pha_loss = [], [], [], [], []
        hemi_wav_loss, large_wav_loss, medium_wav_loss, shoe_wav_loss, small_wav_loss = [], [], [], [], []
        hemi_wav_loss_50ms, large_wav_loss_50ms, medium_wav_loss_50ms, shoe_wav_loss_50ms, small_wav_loss_50ms = [], [], [], [], []
        hemi_missa_amp_loss, large_missa_amp_loss, medium_missa_amp_loss, shoe_missa_amp_loss, small_missa_amp_loss = [], [], [], [], []
        hemi_missa_wav_loss, large_missa_wav_loss, medium_missa_wav_loss, shoe_missa_wav_loss, small_missa_wav_loss = [], [], [], [], []
        hemi_sdr_metric, large_sdr_metric, medium_sdr_metric, shoe_sdr_metric, small_sdr_metric = [], [], [], [], []
        hemi_similarity_metric, large_similarity_metric, medium_similarity_metric, shoe_similarity_metric, small_similarity_metric = [], [], [], [], []

        hemi_count, large_count, medium_count, shoe_count, small_count = 0, 0, 0, 0, 0

        plot_countdown = 0
        plot_count = 0

        progBar = Progbar(numUpdates)

        start = time.time()

        for i in range(0, numUpdates):

            spec_in, emb, spec_out, characteristic = test_generator.__getitem__(i)

            start_inf = time.time()
            if model_name == "unet-vae" or model_name == 'unet-vae-emb':
                spec_generated, _, _ = trained_model.model([spec_in, emb], training=False)
            else:
                spec_generated = trained_model.model([spec_in, emb], training=False)
            end_inf = time.time()

            time_inference.append(end_inf - start_inf)

            for j in range(0, emb.shape[0]):
                start_gen = time.time()

                if diff_gen:
                    diff_phase_generated = (spec_generated[j, :, :, 1] + spec_in[j, :, :, 1]).numpy()
                    diff_spec_generated = np.stack((spec_generated[j, :, :, 0], diff_phase_generated), axis=-1)
                    wav_pred = postprocessor.post_process(diff_spec_generated, emb[j, 1, :], min_max_vector)
                else:
                    wav_pred = postprocessor.post_process(spec_generated[j], emb[j, 1, :], min_max_vector)

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
                    f'{dataset_dir}/room_impulse/{characteristic_out[0]}/Zone{characteristic_out[1]}/{characteristic_out[2]}'
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

                sdr_metric_wav = sdr(wav_pred, wav_true)
                sdr_metric.append(sdr_metric_wav)

                similarity = calculate_similarity(wav_true, wav_pred)
                similarity_metric.append(similarity)

                if characteristic_out[0] == 'HemiAnechoicRoom':
                    hemi_count += 1

                    hemi_total_loss.append(loss)
                    hemi_amp_loss.append(loss_stft)
                    hemi_pha_loss.append(loss_phase)

                    hemi_wav_loss.append(loss_wav)
                    hemi_wav_loss_50ms.append(loss_wav_50ms)

                    hemi_missa_amp_loss.append(loss_missa_amp)
                    hemi_missa_wav_loss.append(loss_missa_wav)

                    hemi_sdr_metric.append(sdr_metric_wav)
                    hemi_similarity_metric.append(similarity)

                if characteristic_out[0] == 'LargeMeetingRoom':
                    large_count += 1

                    large_total_loss.append(loss)
                    large_amp_loss.append(loss_stft)
                    large_pha_loss.append(loss_phase)

                    large_wav_loss.append(loss_wav)
                    large_wav_loss_50ms.append(loss_wav_50ms)

                    large_missa_amp_loss.append(loss_missa_amp)
                    large_missa_wav_loss.append(loss_missa_wav)

                    large_sdr_metric.append(sdr_metric_wav)
                    large_similarity_metric.append(similarity)

                if characteristic_out[0] == 'MediumMeetingRoom':
                    medium_count += 1

                    medium_total_loss.append(loss)
                    medium_amp_loss.append(loss_stft)
                    medium_pha_loss.append(loss_phase)

                    medium_wav_loss.append(loss_wav)
                    medium_wav_loss_50ms.append(loss_wav_50ms)

                    medium_missa_amp_loss.append(loss_missa_amp)
                    medium_missa_wav_loss.append(loss_missa_wav)

                    medium_sdr_metric.append(sdr_metric_wav)
                    medium_similarity_metric.append(similarity)

                if characteristic_out[0] == 'ShoeBoxRoom':
                    shoe_count += 1

                    shoe_total_loss.append(loss)
                    shoe_amp_loss.append(loss_stft)
                    shoe_pha_loss.append(loss_phase)

                    shoe_wav_loss.append(loss_wav)
                    shoe_wav_loss_50ms.append(loss_wav_50ms)

                    shoe_missa_amp_loss.append(loss_missa_amp)
                    shoe_missa_wav_loss.append(loss_missa_wav)

                    shoe_sdr_metric.append(sdr_metric_wav)
                    shoe_similarity_metric.append(similarity)

                if characteristic_out[0] == 'SmallMeetingRoom':
                    small_count += 1

                    small_total_loss.append(loss)
                    small_amp_loss.append(loss_stft)
                    small_pha_loss.append(loss_phase)

                    small_wav_loss.append(loss_wav)
                    small_wav_loss_50ms.append(loss_wav_50ms)

                    small_missa_amp_loss.append(loss_missa_amp)
                    small_missa_wav_loss.append(loss_missa_wav)

                    small_sdr_metric.append(sdr_metric_wav)
                    small_similarity_metric.append(similarity)

                end_loss = time.time()
                time_loss.append(end_loss - start_loss)

                if plot_countdown == 1280:
                    pass
                    create_directory_if_none(f'{saving_path}/{model_name + modifier}_{algorithm}/png/')
                    plot_feature_vs_wav(stft_pred, wav_pred, model_name + modifier, characteristic_out,
                                        f'{saving_path}/{model_name + modifier}_{algorithm}/png/spec_vs_wav_{plot_count}.png')
                    plot_feature_vs_feature_wav(wav_true, stft_true, stft_pred, model_name + modifier,
                                                characteristic_out,
                                                f'{saving_path}/{model_name + modifier}_{algorithm}/png/spec_vs_spec_{plot_count}.png')
                    plot_phase_vs_phase(phase_true, phase_pred, model_name + modifier, characteristic_out,
                                        f'{saving_path}/{model_name + modifier}_{algorithm}/png/phase_vs_phase_{plot_count}.png')
                    plot_wav_vs_wav(wav_true, wav_pred, model_name + modifier, characteristic_out,
                                    f'{saving_path}/{model_name + modifier}_{algorithm}/png/wav_vs_wav_{plot_count}.png')
                    plot_count += 1
                    plot_countdown = 0
                else:
                    plot_countdown += 1

            progBar.update(i)

        progBar.update(test_generator.__len__(), finalize=True)

        end = time.time()
        total_loss = np.mean(total_loss)
        amp_loss = np.mean(amp_loss)
        pha_loss = np.mean(pha_loss)
        wav_loss = np.mean(wav_loss)
        wav_loss_50ms = np.mean(wav_loss_50ms)
        missa_amp_loss = np.mean(missa_amp_loss)
        missa_wav_loss = np.mean(missa_wav_loss)
        sdr_metric = np.mean(sdr_metric)
        similarity_metric = np.mean(similarity_metric)

        hemi_total_loss = np.mean(hemi_total_loss)
        hemi_amp_loss = np.mean(hemi_amp_loss)
        hemi_pha_loss = np.mean(hemi_pha_loss)
        hemi_wav_loss = np.mean(hemi_wav_loss)
        hemi_wav_loss_50ms = np.mean(hemi_wav_loss_50ms)
        hemi_missa_amp_loss = np.mean(hemi_missa_amp_loss)
        hemi_missa_wav_loss = np.mean(hemi_missa_wav_loss)
        hemi_sdr_metric = np.mean(hemi_sdr_metric)
        hemi_similarity_metric = np.mean(hemi_similarity_metric)

        large_total_loss = np.mean(large_total_loss)
        large_amp_loss = np.mean(large_amp_loss)
        large_pha_loss = np.mean(large_pha_loss)
        large_wav_loss = np.mean(large_wav_loss)
        large_wav_loss_50ms = np.mean(large_wav_loss_50ms)
        large_missa_amp_loss = np.mean(large_missa_amp_loss)
        large_missa_wav_loss = np.mean(large_missa_wav_loss)
        large_sdr_metric = np.mean(large_sdr_metric)
        large_similarity_metric = np.mean(large_similarity_metric)

        medium_total_loss = np.mean(medium_total_loss)
        medium_amp_loss = np.mean(medium_amp_loss)
        medium_pha_loss = np.mean(medium_pha_loss)
        medium_wav_loss = np.mean(medium_wav_loss)
        medium_wav_loss_50ms = np.mean(medium_wav_loss_50ms)
        medium_missa_amp_loss = np.mean(medium_missa_amp_loss)
        medium_missa_wav_loss = np.mean(medium_missa_wav_loss)
        medium_sdr_metric = np.mean(medium_sdr_metric)
        medium_similarity_metric = np.mean(medium_similarity_metric)

        shoe_total_loss = np.mean(shoe_total_loss)
        shoe_amp_loss = np.mean(shoe_amp_loss)
        shoe_pha_loss = np.mean(shoe_pha_loss)
        shoe_wav_loss = np.mean(shoe_wav_loss)
        shoe_wav_loss_50ms = np.mean(shoe_wav_loss_50ms)
        shoe_missa_amp_loss = np.mean(shoe_missa_amp_loss)
        shoe_missa_wav_loss = np.mean(shoe_missa_wav_loss)
        shoe_sdr_metric = np.mean(shoe_sdr_metric)
        shoe_similarity_metric = np.mean(shoe_similarity_metric)

        small_total_loss = np.mean(small_total_loss)
        small_amp_loss = np.mean(small_amp_loss)
        small_pha_loss = np.mean(small_pha_loss)
        small_wav_loss = np.mean(small_wav_loss)
        small_wav_loss_50ms = np.mean(small_wav_loss_50ms)
        small_missa_amp_loss = np.mean(small_missa_amp_loss)
        small_missa_wav_loss = np.mean(small_missa_wav_loss)
        small_sdr_metric = np.mean(small_sdr_metric)
        small_similarity_metric = np.mean(small_similarity_metric)

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
                                      np.format_float_scientific(small_missa_wav_loss, precision=4)],
            "SDR": [np.format_float_scientific(sdr_metric, precision=4),
                    np.format_float_scientific(hemi_sdr_metric, precision=4),
                    np.format_float_scientific(large_wav_loss, precision=4),
                    np.format_float_scientific(medium_wav_loss, precision=4),
                    np.format_float_scientific(shoe_wav_loss, precision=4),
                    np.format_float_scientific(small_wav_loss, precision=4)],
            "Similarity": [np.format_float_scientific(similarity_metric, precision=4),
                           np.format_float_scientific(hemi_similarity_metric, precision=4),
                           np.format_float_scientific(large_similarity_metric, precision=4),
                           np.format_float_scientific(medium_similarity_metric, precision=4),
                           np.format_float_scientific(shoe_similarity_metric, precision=4),
                           np.format_float_scientific(small_similarity_metric, precision=4)]
        }

        time_dataframe = pd.DataFrame(time_data)
        loss_dataframe = pd.DataFrame(loss_data)

        time_dataframe.to_csv(
            f'{saving_path}/{model_name + modifier}_{algorithm}/{model_name + modifier}_infer_time.csv', index=False)
        loss_dataframe.to_csv(f'{saving_path}/{model_name + modifier}_{algorithm}/{model_name + modifier}_losses.csv',
                              index=False)

        print('Done! Clearing cache and allocated memory')

    del trained_model
    K.clear_session()

    del dataset
    del test_generator
