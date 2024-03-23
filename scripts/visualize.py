import math
import os
import pathlib
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf


def create_directory_if_none(dir_path):
    """
    Creates a directory.

    :param dir_path: Path to make directory.
    """
    dir_path = dir_path
    directory = pathlib.Path(dir_path)
    if not directory.exists():
        os.makedirs(dir_path)

def plot_wav(signal):

    time_ax = np.linspace(0, len(signal) / 48000, num=len(signal))
    plt.plot(time_ax, signal)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

def plot_spec(stft):
    if len(stft.shape) > 2:
        assert len(stft.shape) == 3
        stft = np.squeeze(stft, axis=-1)
    height = stft.shape[0]
    width = stft.shape[1]
    x = np.linspace(0, np.size(stft), num=width, dtype=int)
    y = range(height)
    plt.pcolormesh(x, y, stft)
    plt.colorbar()
    plt.show()

def plot_feature_vs_wav(stft, signal, model, characteristics, path):
    fig, ax = plt.subplots(2, figsize=(12, 8))
    if len(stft.shape) > 2:
        assert len(stft.shape) == 3
        stft = np.squeeze(stft, axis=-1)
    stft = stft
    height = stft.shape[0]
    width = stft.shape[1]
    x = np.linspace(0, np.size(stft), num=width, dtype=int)
    y = range(height)
    ax[1].pcolormesh(x, y, stft)
    ax[1].set_title('Spectogram')

    time_ax = np.linspace(0, len(signal) / 48000, num=len(signal))
    ax[0].plot(time_ax, signal)
    ax[0].set_title('Wav')

    fig.suptitle(f'Model {model}: {characteristics[0]} {characteristics[1]} {characteristics[2]} L{characteristics[3]} M{characteristics[4]}')
    plt.savefig(path)
    plt.close(fig)


def plot_feature_vs_feature_wav(signal, stft_true, stft_pred, model, characteristics, path):

    fig, ax = plt.subplots(3, figsize=(12, 12))
    time_ax = np.linspace(0, len(signal) / 48000, num=len(signal))
    ax[0].plot(time_ax, signal)
    ax[0].set_title('Wav true')

    stft_true = stft_true
    if len(stft_true .shape) > 2:
        assert len(stft_true .shape) == 3
        stft_true = np.squeeze(stft_true , axis=-1)
    stft = stft_true
    height = stft.shape[0]
    width = stft.shape[1]
    x = np.linspace(0, np.size(stft), num=width, dtype=int)
    y = range(height)
    ax[1].pcolormesh(x, y, stft)
    ax[1].set_title('Spectogram true')

    stft_pred = stft_pred
    if len(stft_pred.shape) > 2:
        assert len(stft_pred.shape) == 3
        stft_pred = np.squeeze(stft_pred, axis=-1)
    stft = stft_pred
    height = stft.shape[0]
    width = stft.shape[1]
    x = np.linspace(0, np.size(stft), num=width, dtype=int)
    y = range(height)
    ax[2].pcolormesh(x, y, stft)
    ax[2].set_title('Spectogram pred')

    fig.suptitle(f'Model {model}: {characteristics[0]} {characteristics[1]} {characteristics[2]} L{characteristics[3]} M{characteristics[4]}')
    plt.savefig(path)
    plt.close(fig)


def plot_phase_vs_phase(phase_true, phase_pred, model, characteristics, path):

    fig, ax = plt.subplots(2, figsize=(12, 8))

    phase_true = phase_true
    if len(phase_true .shape) > 2:
        assert len(phase_true .shape) == 3
        phase_true = np.squeeze(phase_true , axis=-1)
    phase = phase_true
    height = phase.shape[0]
    width = phase.shape[1]
    x = np.linspace(0, np.size(phase), num=width, dtype=int)
    y = range(height)
    ax[0].pcolormesh(x, y, phase)
    ax[0].set_title('Phase true')

    phase_pred = phase_pred
    if len(phase_pred.shape) > 2:
        assert len(phase_pred.shape) == 3
        phase_pred = np.squeeze(phase_pred, axis=-1)
    phase = phase_pred
    height = phase.shape[0]
    width = phase.shape[1]
    x = np.linspace(0, np.size(phase), num=width, dtype=int)
    y = range(height)
    ax[1].pcolormesh(x, y, phase)
    ax[1].set_title('Phase pred')

    fig.suptitle(f'Model {model}: {characteristics[0]} {characteristics[1]} {characteristics[2]} L{characteristics[3]} M{characteristics[4]}')
    plt.savefig(path)
    plt.close(fig)


def plot_wav_vs_wav(signal_true, signal_pred, model, characteristics, path):

    fig, ax = plt.subplots(2, figsize=(12, 8))

    time_ax = np.linspace(0, len(signal_true) / 48000, num=len(signal_true))
    ax[0].plot(time_ax, signal_true)
    ax[0].set_title('Wav true')

    time_ax = np.linspace(0, len(signal_pred) / 48000, num=len(signal_pred))
    ax[1].plot(time_ax, signal_pred)
    ax[1].set_title('Wav pred')

    fig.suptitle(f'Model {model}: {characteristics[0]} {characteristics[1]} {characteristics[2]} L{characteristics[3]} M{characteristics[4]}')
    plt.savefig(path)
    plt.close(fig)


def sigmoid(beta, dimensions):
    x = np.linspace(-10, 10, dimensions[1])
    z = 1 / (1 + np.exp(-(x) * beta))
    z = np.flip(z)
    sig = np.tile(z, (dimensions[0], 1))

    plt.plot(x, sig[0,:])
    plt.xlabel("x")
    plt.ylabel("Sigmoid(X)")

    plt.show()

    return sig


if __name__ == '__main__':

    sig = sigmoid(1, (144, 160))
    sig = np.repeat(np.expand_dims(sig, 0), 16, axis=0)
    print(sig.shape)
    phase = 0
    phases = (phase + math.pi) % (2 * math.pi) - math.pi
    print(phases)

    # loader = Loader(48000, 0.9, True)
    # signal = loader.load("../../datasets/room_impulse/LargeMeetingRoom\ZoneA\PlanarMicrophoneArray\LargeMeetingRoom_ZoneA_PlanarMicrophoneArray_L1_M1.wav")
    # time_ax = np.linspace(0, len(signal) / 48000, num=len(signal))
    # plt.plot(time_ax, signal)
