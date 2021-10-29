import torch
from VoiceClassification import VoiceClassification
from torch.utils.data import DataLoader
from voiceDataset import VoiceDataset, pad_collate

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def train_loop(model, dataloader):
    for batch, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        pred = model(images)
        print(pred.size())
        arg_maxes = torch.argmax(pred, dim=2)
        for i, args in enumerate(arg_maxes):
            for j, index in enumerate(args):
                print(j, index.size())


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
        "batch_size": 2,
        "epochs": 1
    }
    # dataset
    training_data = VoiceDataset(path="../dataset/voice/data_thchs30", train=True)
    test_data = VoiceDataset(path="../dataset/voice/data_thchs30", train=False)
    train_dataloader = DataLoader(training_data, batch_size=params['batch_size'], collate_fn=lambda b: pad_collate(b),
                                  shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=params['batch_size'], collate_fn=lambda b: pad_collate(b),
                                 shuffle=False)
    # model
    myModel = VoiceClassification(params['n_cnn_layers'], params['n_rnn_layers'], params['rnn_dim'],
                                  params['n_class'], params['n_feats'], params['stride'], params['dropout']).to(device)
    train_loop(myModel, train_dataloader)
    torch.save(myModel, '../param/voice.pth')
