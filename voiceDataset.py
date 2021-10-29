import os
from typing import Tuple
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from encodeAndDecode import Encode


def load_voice_item(filename: str,
                    path: str) -> Tuple[Tensor, int, Tensor]:
    filename = path + "/data/" + filename
    waveform, sample_rate = torchaudio.load(filename)
    waveform = torch.flatten(waveform)
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
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=20,
            melkwargs={
                'n_fft': 2048,
                'n_mels': 256,
                'hop_length': 512,
                'mel_scale': 'htk',
            }
        )
        label_lengths.append(len(label))
        labels.append(label)
        mfcc.append(mfcc_transform(waveform).transpose(0, 1))
        input_lengths.append(mfcc[-1].shape[0] // 2)
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

    def __getitem__(self, idx) -> Tuple[Tensor, int, Tensor]:
        filename = self._walker[idx]
        return load_voice_item(filename, self._dir)

    def __len__(self) -> int:
        return len(self._walker)
