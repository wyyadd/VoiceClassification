import torch
import torch.nn as nn
from VoiceClassificationModel import VoiceClassificationModel
from torch.utils.data import DataLoader
from voiceDataset import VoiceDataset, pad_collate

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def train_loop(model, dataloader, loss_function, optimizer, scheduler, epoch):
    data_len = len(dataloader.dataset)
    for batch, (spectrogram, labels, input_lengths, label_lengths) in enumerate(dataloader):
        spectrogram, labels = spectrogram.to(device), labels.to(device)

        pred = model(spectrogram)
        pred.transpose(0, 1)
        loss = loss_function(pred, labels, input_lengths, label_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if batch % 100 == 0 or batch == data_len:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch * len(spectrogram), data_len,
                       100. * batch / len(dataloader), loss.item()))


if __name__ == "__main__":
    params = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 220,
        "n_feats": 256,
        "stride": 2,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "batch_size": 1,
        "epochs": 3
    }
    # dataset
    training_data = VoiceDataset(path="../dataset/voice/data_thchs30", train=True)
    test_data = VoiceDataset(path="../dataset/voice/data_thchs30", train=False)
    train_dataloader = DataLoader(training_data, batch_size=params['batch_size'], collate_fn=lambda b: pad_collate(b),
                                  shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=params['batch_size'], collate_fn=lambda b: pad_collate(b),
                                 shuffle=False)
    # model
    myModel = VoiceClassificationModel(params['n_cnn_layers'], params['n_rnn_layers'], params['rnn_dim'],
                                       params['n_class'], params['n_feats'], params['stride'], params['dropout']).to(
        device)
    # loss_fn and optimizer
    opt = torch.optim.AdamW(myModel.parameters(), params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt,
                                                    max_lr=params['learning_rate'],
                                                    steps_per_epoch=int(len(train_dataloader)),
                                                    epochs=params['epochs'],
                                                    anneal_strategy='linear')
    loss_fn = nn.CTCLoss(blank=220).to(device)
    for epoch in range(1, params["epochs"]+1):
        train_loop(myModel, train_dataloader, loss_fn, opt, scheduler, epoch)
    torch.save(myModel, '../param/voice.pth')
 #