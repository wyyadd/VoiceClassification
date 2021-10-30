import torch
import torchaudio
import torch.nn.functional as F
import encodeAndDecode


def test():
    model_path = ""
    filename = ""
    model = torch.load(model_path)
    waveform, sample_rate = torchaudio.load(filename)
    waveform = torch.flatten(waveform)
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=256,
        melkwargs={
            'n_fft': 2048,
            'n_mels': 256,
            'hop_length': 512,
            'mel_scale': 'htk',
        }
    )
    # batch=1, channel=1, feature, time
    voice = mfcc_transform(waveform).unsqueeze(0).unsqueeze(0)
    # batch, time, n_class
    pred = model(voice)
    pred = F.log_softmax(pred, dim=2)
    decoded_preds, _ = encodeAndDecode.decode.greed_decode(pred)
    chinese = encodeAndDecode.decode.pinyin2chinese(decoded_preds[0])
    return chinese


print(test())

