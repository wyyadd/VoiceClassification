import os
from typing import Tuple
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from encodeAndDecode import Encode


# 根据wav文件地址提取音频
def load_voice_item(filename: str,
                    path: str) -> Tuple[Tensor, int, Tensor, str]:
    filename = path + "/data/" + filename
    # 音频wave， 采样率
    waveform, sample_rate = torchaudio.load(filename, normalize=True)
    # 转为1维
    waveform = torch.flatten(waveform)
    pinyin_label = []
    chinese_label = str
    with open(filename + ".trn", encoding="utf8") as labels:
        lines = labels.readlines()
        # 中文标签
        chinese_label = lines[0].strip().replace(' ', '')
        # 拼音标签
        pinyin_label = lines[2].strip().split(' ')
    encode = Encode()
    pinyin_label = torch.tensor(encode.text_to_int(pinyin_label))
    return waveform, sample_rate, pinyin_label, chinese_label


# 同时mfcc提取音频特征
# 由于音频长度不同， 将不同tensor转为相同size的tensor(长度统一为最长的tensor)
def pad_collate(batch):
    mfcc = []
    pinyin_labels = []
    input_lengths = []
    label_lengths = []
    chinese_labels = []
    for waveform, sample_rate, pinyin_label, chinese_label in batch:
        # mfcc参数
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            # Number of mfc coefficients to retain
            n_mfcc=40,
            melkwargs={
                # Size of FFT
                'n_fft': 2048,
                # Number of mel filterbanks
                'n_mels': 128,
                # Length of hop between STFT windows
                'hop_length': 512,
                #  Scale to use: htk or slaney
                'mel_scale': 'htk',
            }
        )
        label_lengths.append(len(pinyin_label))
        pinyin_labels.append(pinyin_label)
        chinese_labels.append(chinese_label)
        # mfcc = [time, feature], transpose = [feature, time]
        mfcc.append(mfcc_transform(waveform).transpose(0, 1))
        input_lengths.append(mfcc[-1].shape[0] // 2)
    # mfcc = [batch channel feature time]
    # 将不同长度mfcc打包成相同长度
    mfcc = torch.nn.utils.rnn.pad_sequence(mfcc, batch_first=True).unsqueeze(1).transpose(2, 3)
    pinyin_labels = torch.nn.utils.rnn.pad_sequence(pinyin_labels, batch_first=True)
    return mfcc, pinyin_labels, input_lengths, label_lengths, chinese_labels


# 根据官方文档，自定义数据集
class VoiceDataset(Dataset):
    def __init__(self, path, train: bool) -> None:
        self._dir = os.path.abspath(path)
        # wav文件地址list
        self._walker = []
        for name in os.listdir(self._dir + ("/train" if train else "/test")):
            if os.path.splitext(name)[1] == ".wav":
                self._walker.append(name)

    def __getitem__(self, idx) -> Tuple[Tensor, int, Tensor, str]:
        filename = self._walker[idx]
        return load_voice_item(filename, self._dir)

    def __len__(self) -> int:
        return len(self._walker)
