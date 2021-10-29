import torch
from torch import nn


class CNNLayerNorm(nn.Module):
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        x = x.transpose(2, 3).contiguous()
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()


class ResidualCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()
        self.Conv1 = nn.Sequential(
            CNNLayerNorm(n_feats),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2)
        )
        self.Conv2 = nn.Sequential(
            CNNLayerNorm(n_feats),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel // 2)
        )

    def forward(self, x):
        residual = x
        x = self.Conv1(x)
        x = self.Conv2(x)
        x += residual
        return x


class GRU(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(GRU, self).__init__()
        self.BiGRU = nn.GRU(input_size=rnn_dim, hidden_size=hidden_size, num_layers=1, batch_first=batch_first,
                            bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.activation(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class VoiceClassification(nn.Module):
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(VoiceClassification, self).__init__()
        n_feats = n_feats // 2
        self.cnn = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=stride, padding=3 // 2)
        self.resCnn_layers = nn.Sequential(
            *[
                ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
                for _ in range(n_cnn_layers)
            ]
        )
        self.fully_connected = nn.Linear(n_feats * 32, rnn_dim)
        self.biRnn_layers = nn.Sequential(*[
            GRU(rnn_dim=rnn_dim if i == 0 else rnn_dim * 2,
                hidden_size=rnn_dim, dropout=dropout, batch_first=i == 0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim * 2, rnn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.cnn(x)
        x = self.resCnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        x = x.transpose(1, 2)
        x = self.fully_connected(x)
        x = self.biRnn_layers(x)
        x = self.classifier(x)
        x = self.softmax(x)
        return x
