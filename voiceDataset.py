import os
from typing import Tuple
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from encodeAndDecode import Encode


def load_voice_item(filename: str,
                    path: str) -> Tuple[Tensor, int, Tensor, str]:
    filename = path + "/data/" + filename
    waveform, sample_rate = torchaudio.load(filename)
    waveform = torch.flatten(waveform)
    pinyin_label = []
    chinese_label = str
    with open(filename + ".trn", encoding="utf8") as labels:
        lines = labels.readlines()
        chinese_label = lines[0].strip().replace(' ', '')
        pinyin_label = lines[2].strip().split(' ')
    encode = Encode()
    pinyin_label = torch.tensor(encode.text_to_int(pinyin_label))
    return waveform, sample_rate, pinyin_label, chinese_label


def pad_collate(batch):
    mfcc = []
    pinyin_labels = []
    input_lengths = []
    label_lengths = []
    chinese_labels = []
    for waveform, sample_rate, pinyin_label, chinese_label in batch:
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=40,
            melkwargs={
                'n_fft': 2048,
                'n_mels': 128,
                'hop_length': 512,
                'mel_scale': 'htk',
            }
        )
        label_lengths.append(len(pinyin_label))
        pinyin_labels.append(pinyin_label)
        chinese_labels.append(chinese_label)
        # time feature
        mfcc.append(mfcc_transform(waveform).transpose(0, 1))
        input_lengths.append(mfcc[-1].shape[0] // 2)
    # batch channel feature time
    mfcc = torch.nn.utils.rnn.pad_sequence(mfcc, batch_first=True).unsqueeze(1).transpose(2, 3)
    pinyin_labels = torch.nn.utils.rnn.pad_sequence(pinyin_labels, batch_first=True)
    return mfcc, pinyin_labels, input_lengths, label_lengths, chinese_labels


class VoiceDataset(Dataset):
    def __init__(self, path, train: bool) -> None:
        self._dir = os.path.abspath(path)
        self._walker = []
        for name in os.listdir(self._dir + ("/train" if train else "/test")):
            if os.path.splitext(name)[1] == ".wav":
                self._walker.append(name)

    def __getitem__(self, idx) -> Tuple[Tensor, int, Tensor, str]:
        filename = self._walker[idx]
        return load_voice_item(filename, self._dir)

    def __len__(self) -> int:
        return len(self._walker)
