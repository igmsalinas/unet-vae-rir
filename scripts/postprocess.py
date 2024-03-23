
"""
Ignacio Martín 2022

Code corresponding to the Bachelor Thesis "Synthesis of Room Impulse Responses by Means of Deep Learning"

Authors:
    Ignacio Martín
    José Antonio Belloch
    Gema Piñero

University Carlos III de Madrid
"""

import os
import pathlib

import librosa
import tensorflow as tf
import numpy as np
from scipy.io.wavfile import write

from preprocess import Normalizer, TensorPadder


class PostProcess:
    """
    PostProcess takes a feature (Log Spectrogram and Phase) normalized and padded,
    deletes the padding and denormalizes it, computes the inverse STFT and converts it into a wav
    Steps:
    1- Loads feature and vector
    2- Deletes padding
    3- Denormalizes them
    4- Log Spec -> Spec
    5- ISTFT
    6- Converts to wav
    """

    def __init__(self, folder, algorithm="ph", n_iters=32, momentum=0, normalize_vector=False):

        self.stft = None
        self.phase = None
        self.waveform = None

        self.normalizer = Normalizer()
        self.padder = TensorPadder((144, 160))

        self.algorithm = algorithm

        self.n_iters = n_iters
        self.momentum = momentum

        self.normalize_vector = normalize_vector

        self.wav_path = folder + f'_{self.algorithm}'

        self._create_directory_if_none(self.wav_path)

    def post_process(self, feature, vector, min_max_vector, des_shape=(129, 151),
                     n_fft=256, win_length=128, hop_length=64, sr=48000):
        """
        Takes a feature, its corresponding vector, the min_max normalization values, the previous shape of the STFT
        and proceeds to transform, denormalize, perform the ISTFT with the given parameters and saves the wav and STFT.

        :param feature: np.ndarray containing log magnitude and phase
        :param vector: vector list of the feature
        :param min_max_vector: min max values used for normalization
        :param des_shape: previous shape before padding
        :param hop_length: hop length stft
        :param win_length: window length stft
        :param n_fft: number of ffts stft
        :param sr: sample rate
        """
        stft, phase = self.get_stft_phase(feature)
        stft_d, phase_d = self.de_shape(stft, phase, des_shape)
        denorm_f, denorm_p = self.denormalize(stft_d, phase_d)
        self.istft(denorm_f, denorm_p, n_fft, win_length, hop_length)
        self.align_wav(vector, min_max_vector, sr)
        # self.save_wav(sr, vector)
        # self.save_stft(feature)

        return self.waveform

    @staticmethod
    def get_stft_phase(feature):
        """
        Unstacks the feature.

        :param feature: input feature
        :return: log stft, phase
        """
        # stft, phase = np.moveaxis(feature, 2, 0)
        stft = feature[:, :, 0]
        phase = feature[:, :, 1]
        return stft, phase

    def de_shape(self, stft, phase, des_shape):
        """
        Removes the padding.

        :param stft: log magnitude stft
        :param phase: phase stft
        :param des_shape: previous shape
        :return: transformed log_stft, phase
        """
        stft_d, phase_d = self.padder.un_pad(stft, phase, des_shape)

        return stft_d, phase_d

    def denormalize(self, stft, phase):
        """
        Denormalizes the log magnitude and phase given the previous min_max.

        :param stft: input log magnitude
        :param phase: input phase
        :param min_max: min max used in normalization
        :return: denormalized feature
        """
        denorm_stft, denorm_phase = self.normalizer.denormalize(stft, phase)
        return denorm_stft, denorm_phase

    def istft(self, denorm_f, denorm_p, n_fft, win_length, hop_length):
        """
        Performs the inverse STFT given the parameters and the denormalized feature.

        :param denorm_f: denormalized magnitude
        :param denorm_p: denormalized phase
        :param hop_length: hop length
        :param win_length: window length
        :param n_fft: number of ffts
        """
        conv_stft = denorm_f * (np.cos(denorm_p) + 1j * np.sin(denorm_p))
        if self.algorithm == 'ph':
            waveform = librosa.istft(conv_stft, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        elif self.algorithm == 'gl_ph':
            waveform = librosa.griffinlim(conv_stft, n_iter=self.n_iters,
                                          n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                                          init=None, momentum=self.momentum)
        elif self.algorithm == 'gl_mag':
            waveform = librosa.griffinlim(denorm_f, n_iter=self.n_iters,
                                          n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                                          momentum=self.momentum)

        self.waveform = waveform

    def save_wav(self, sr, vector):
        """
        Writes wav given a sample rate and the vector corresponding to the waveform.

        :param sr: sample rate
        :param vector: information vector
        """
        vector_name = ""
        for value in vector:
            value_str = str(value)
            vector_name += "-" + value_str
        self.wav_name = "RIR" + vector_name
        self._create_directory_if_none(self.wav_path + "/rir/")
        file_path = os.path.join(self.wav_path + "/rir/", self.wav_name + ".wav")
        write(file_path, sr, self.waveform)

    def save_stft(self, feature):
        """
        Saves the log magnitude and phase generated by the model.

        :param feature: generated stft
        """
        self._create_directory_if_none(self.wav_path + "/stft/")
        file_path = os.path.join(self.wav_path + "/stft/", self.wav_name)
        np.save(file_path + ".npy", feature)

    @staticmethod
    def _create_directory_if_none(dir_path):
        """
        Creates a directory.

        :param dir_path: Path to make directory.
        """
        dir_path = dir_path
        directory = pathlib.Path(dir_path)
        if not directory.exists():
            os.makedirs(dir_path)

    def denorm_embedding(self, emb, min_max_vector):
        """
        Denormalizes the embedding.
        """
        denorm_emb = self.normalizer.denormalize_embedding(emb, min_max_vector)
        return denorm_emb

    @staticmethod
    def find_direct_sound_index(waveform):
        """
        Find the direct sound index in the waveform. Its either the first maximum of the first minimum.
        waveform: The input waveform array.
        """
        # Find the first maximum and minimum
        max_index = np.argmax(waveform)
        min_index = np.argmin(waveform)
        # Return the index of the first maximum or minimum and if it is the maximum or the minimum
        if max_index < min_index:
            return True, max_index
        else:
            return False, min_index


    def align_wav(self, emb, min_max_vector, sr=48000):
        """
        1. Obtain direct sound from embedding
        2. Align the waveform with the direct sound
        3. Set to 0 the samples previous to the direct sound
        """

        # Your existing code to normalize and calculate distances
        if self.normalize_vector:
            emb = self.denorm_embedding(emb, min_max_vector)
        listener_pos = emb[9:12]
        speaker_pos = emb[12:15]
        # Obtain the distance between the listener and the speaker
        distance = np.sqrt(np.sum((listener_pos - speaker_pos) ** 2))
        # Calculate the number of samples to set to 0 given the distance and the speed of sound at 20 degrees and the sampling rate
        num_samples = int((distance / 34300) * sr)
        original_length = len(self.waveform)
        # Find the direct sound index
        min_max, direct_sound_index = self.find_direct_sound_index(self.waveform)
        # Get the sum of previous samples before the direct sound
        sum_previous_samples = np.sum(np.abs(self.waveform[:direct_sound_index]))

        # Add or subtract the sum of previous samples to the samples that have been set to 0
        # if min_max:
        #     self.waveform[direct_sound_index] += sum_previous_samples
        # else:
        #     self.waveform[direct_sound_index] -= sum_previous_samples

        # Align the waveform with the direct sound
        self.waveform = np.concatenate((np.zeros(num_samples), self.waveform[direct_sound_index:]), dtype=np.float32)
        if len(self.waveform) > original_length:
            self.waveform = self.waveform[:original_length]
        elif len(self.waveform) < original_length:
            self.waveform = np.concatenate((self.waveform, np.zeros(original_length - len(self.waveform))), dtype=np.float32)
        # Return the aligned waveform
        return self.waveform



