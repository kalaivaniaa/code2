import torch
import torch.nn as nn
import torch.nn.functional as F

class MHA_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_heads):
        super(MHA_LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.local_cnn = nn.Conv1d(embedding_dim, 128, kernel_size=3, padding=1)
        self.global_cnn = nn.Conv1d(embedding_dim, 128, kernel_size=6, padding=3)
        self.lstm = nn.LSTM(256, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        local_feat = F.relu(self.local_cnn(x))
        global_feat = F.relu(self.global_cnn(x))
        feat = torch.cat((local_feat, global_feat), dim=1)
        feat = feat.permute(0, 2, 1)
        lstm_out, _ = self.lstm(feat)
        attn_output, _ = self.attn(lstm_out, lstm_out, lstm_out)
        out = self.fc(attn_output[:, -1, :])
        return torch.sigmoid(out)
