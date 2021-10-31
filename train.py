import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from VoiceClassificationModel import VoiceClassificationModel
from voiceDataset import VoiceDataset, pad_collate
import encodeAndDecode

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def train_loop(model, dataloader, loss_function, optimizer, scheduler, epoch):
    print("-----start train----")
    data_len = len(dataloader.dataset)
    model.train()
    for batch, (spectrogram, pinyin_labels, input_lengths, label_lengths, _) in enumerate(dataloader):
        spectrogram, pinyin_labels = spectrogram.to(device), pinyin_labels.to(device)
        # batch, time, n_class
        pred = model(spectrogram)
        pred = F.log_softmax(pred, dim=2)
        # time, batch, n_class
        pred = pred.transpose(0, 1)
        loss = loss_function(pred, pinyin_labels, input_lengths, label_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()
        if batch % 100 == 0 or batch == data_len:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch * len(spectrogram), data_len,
                       100. * batch / len(dataloader), loss.item()))


def test_loop(model, dataloader, loss_function):
    print("-----start evaluate----")
    test_loss = 0
    test_cer, test_wer = [], []
    model.eval()
    with torch.no_grad():
        for index, (spectrogram, pinyin_labels, input_lengths, label_lengths, chinese_labels) in enumerate(dataloader):
            if index == 10:
                break
            spectrogram, pinyin_labels = spectrogram.to(device), pinyin_labels.to(device)
            # batch, time, n_class
            pred = model(spectrogram)
            pred = F.log_softmax(pred, dim=2)
            # time, batch, n_class
            pred = pred.transpose(0, 1)
            loss = loss_function(pred, pinyin_labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(dataloader)
            decoded_preds, decoded_targets = encodeAndDecode.decode.greed_decode(pred.transpose(0, 1), pinyin_labels,
                                                                                 label_lengths)
            for j in range(len(decoded_preds)):
                target = chinese_labels[j]
                pred_str = encodeAndDecode.decode.pinyin2chinese(decoded_preds[j])
                test_cer.append(encodeAndDecode.cer(target, pred_str))
                test_wer.append(encodeAndDecode.wer(target, pred_str))
                if index % 9 == 0 and index != 0:
                    print('Predict: {} \n target: {}'.format(pred_str,
                                                             target))

    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)
    print(
        'Test set: Average loss: {:.4f}, Average CER: {:4f}, Average WER: {:4f}\n'.format(test_loss, avg_cer, avg_wer))


if __name__ == "__main__":
    params = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 220,
        "n_feats": 40,
        "stride": 2,
        "dropout": 0.1,
        "learning_rate": 5e-4,
        "batch_size": 20,
        "epochs": 1
    }
    # dataset
    training_data = VoiceDataset(path="../dataset/voice/data_thchs30", train=True)
    test_data = VoiceDataset(path="../dataset/voice/data_thchs30", train=False)
    train_dataloader = DataLoader(training_data, batch_size=params['batch_size'], collate_fn=lambda b: pad_collate(b),
                                  shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=params['batch_size'], collate_fn=lambda b: pad_collate(b),
                                 shuffle=True)
    # model
    myModel = VoiceClassificationModel(params['n_cnn_layers'], params['n_rnn_layers'], params['rnn_dim'],
                                       params['n_class'], params['n_feats'], params['stride'], params['dropout']).to(
        device)
    # myModel = torch.load('../param/voice_nnf_40.pth')
    # loss_fn and optimizer
    opt = torch.optim.AdamW(myModel.parameters(), params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt,
                                                    max_lr=params['learning_rate'],
                                                    steps_per_epoch=int(len(train_dataloader)),
                                                    epochs=params['epochs'],
                                                    anneal_strategy='linear')
    loss_fn = nn.CTCLoss(blank=0).to(device)
    # train and test
    for epoch in range(1, params["epochs"] + 1):
        train_loop(myModel, train_dataloader, loss_fn, opt, scheduler, epoch)
        test_loop(myModel, test_dataloader, loss_fn)
    torch.save(myModel, '../param/voice_nnf_40_new.pth')
