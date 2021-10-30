import torch
import torchaudio
import torch.nn.functional as F
import encodeAndDecode

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def test():
    model_path = "../param/voice_nnf_20.pth"
    filename = "E:/FFOutput/test.wav"
    # model
    myModel = torch.load(model_path)
    waveform, sample_rate = torchaudio.load(filename)
    waveform = torch.flatten(waveform)
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=20,
        melkwargs={
            'n_fft': 2048,
            'n_mels': 128,
            'hop_length': 512,
            'mel_scale': 'htk'
        }
    )
    # batch=1, channel=1, feature, time
    voice = mfcc_transform(waveform).unsqueeze(0).unsqueeze(0)
    voice = voice.to(device)
    # batch, time, n_class
    pred = myModel(voice)
    pred = F.log_softmax(pred, dim=2)
    decoded_preds, _ = encodeAndDecode.decode.greed_decode(pred)
    print(decoded_preds[0])
    chinese = encodeAndDecode.decode.pinyin2chinese(decoded_preds[0])
    return chinese

print(test())

