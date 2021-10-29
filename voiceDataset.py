import os
from typing import Tuple
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from encodeAndDecode import Encode
import librosa
import numpy


def load_voice_item(filename: str,
                    path: str) -> Tuple[numpy.ndarray, int, Tensor]:
    filename = path + "/data/" + filename
    waveform, sample_rate = librosa.load(filename)
    label = []
    with open(filename + ".trn", encoding="utf8") as labels:
        for line in labels:
            label = line.strip()
    label = label.split(' ')
    encode = Encode()
    label = torch.tensor(encode.text_to_int(label))
    return waveform, sample_rate, label


def pad_collate(batch):
    mfcc = []
    labels = []
    input_lengths = []
    label_lengths = []
    for waveform, sample_rate, label in batch:
        label_lengths.append(len(label))
        labels.append(label)
        # feature, time
        feature = librosa.feature.mfcc(waveform, sample_rate, n_mfcc=40)
        # time, feature
        mfcc.append(torch.from_numpy(feature).transpose(0, 1))
        input_lengths.append(len(feature[1]) // 2)
    mfcc = torch.nn.utils.rnn.pad_sequence(mfcc, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
    return mfcc, labels, input_lengths, label_lengths


class VoiceDataset(Dataset):
    def __init__(self, path, train: bool) -> None:
        self._dir = os.path.abspath(path)
        self._walker = []
        for name in os.listdir(self._dir + ("/train" if train else "/test")):
            if os.path.splitext(name)[1] == ".wav":
                self._walker.append(name)

    def __getitem__(self, idx) -> Tuple[numpy.ndarray, int, Tensor]:
        filename = self._walker[idx]
        return load_voice_item(filename, self._dir)

    def __len__(self) -> int:
        return len(self._walker)
