import numpy as np
import random
from tensorflow.keras.utils import Sequence

from dataset import Dataset


class DataGenerator(Sequence):

    def __init__(self, dataset, batch_size=32, partition='train', shuffle=True, characteristics=False,
                 normalize_vector=False):

        'Initialization'
        self.dataset = dataset
        self.batch_size = batch_size
        self.partition = partition
        self.shuffle = shuffle
        self.characteristics = characteristics
        self.normalize_vector = normalize_vector

        'Secondary Initializations'
        self._idx = 0
        self.index_in = dataset.index_in
        self.index_out = dataset.index_out


        temp = list(zip(self.index_in, self.index_out))
        random.Random(dataset.seed).shuffle(temp)
        index_in, index_out = zip(*temp)

        self.index_in = list(index_in)
        self.index_out = list(index_out)

        self.characteristics_list = self.dataset.return_characteristics()

        if partition == 'train':
            self.index_in = self.index_in[:int(0.7 * len(self.index_in))]
            self.index_out = self.index_out[:int(0.7 * len(self.index_out))]
        elif partition == 'val':
            self.index_in = self.index_in[int(0.7 * len(self.index_in)):int(0.9 * len(self.index_in))]
            self.index_out = self.index_out[int(0.7 * len(self.index_out)):int(0.9 * len(self.index_out))]
        elif partition == 'test':
            self.index_in = self.index_in[int(0.9 * len(self.index_in)):]
            self.index_out = self.index_out[int(0.9 * len(self.index_out)):]

    def __len__(self):

        N = len(self.index_in)
        b = self.batch_size
        return int(N // b)

    def __iter__(self):

        return self

    def on_epoch_end(self):

        if self.shuffle:
            temp = list(zip(self.index_in, self.index_out))
            random.shuffle(temp)
            index_in, index_out = zip(*temp)
            self.index_in = list(index_in)
            self.index_out = list(index_out)

    def __getitem__(self, idx):

        # Load images and include into the batch (inputs)
        stft_in, phase_in, emb_in, char_in = [], [], [], []
        for i in np.arange(idx * self.batch_size, (idx + 1) * self.batch_size):
            stft, phase, emb = self.dataset.__getitem__(self.index_in[int(i)])
            stft_in.append(stft)
            phase_in.append(phase)
            emb_in.append(emb)

            if self.characteristics:
                char_in.append(self.characteristics_list[self.index_in[int(i)]])

        # Load images and include into the batch (outputs)
        stft_out, phase_out, emb_out, char_out = [], [], [], []
        for i in np.arange(idx * self.batch_size, (idx + 1) * self.batch_size):
            stft, phase, emb = self.dataset.__getitem__(self.index_out[int(i)])
            stft_out.append(stft)
            phase_out.append(phase)
            emb_out.append(emb)

            if self.characteristics:
                char_out.append(self.characteristics_list[self.index_out[int(i)]])

        spectrogram_in = np.stack((stft_in, phase_in), axis=-1)
        spectrogram_out = np.stack((stft_out, phase_out), axis=-1)

        embedding = np.stack((emb_in, emb_out), axis=1)

        if self.normalize_vector:
            dtype = 'float32'
        else:
            dtype = 'int32'

        if self.characteristics:
            characteristic = np.stack((char_in, char_out), axis=2)
            return np.array(spectrogram_in).astype('float32'), np.array(embedding).astype(dtype), \
                   np.array(spectrogram_out).astype('float32'), characteristic
        else:
            return np.array(spectrogram_in).astype('float32'), np.array(embedding).astype(dtype), \
                   np.array(spectrogram_out).astype('float32')


if __name__ == '__main__':

    dataset = Dataset('../../../datasets', 'room_impulse', normalization=True, debugging=True, extract=False,
                      room=['All'])

    train_generator = DataGenerator(dataset, batch_size=16, partition='train', shuffle=True)