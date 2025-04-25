import torch.nn as nn

class EmotionClassifierLSTM(nn.Module):
    def __init__(self, vocab_size: int, emb_size:int, hidden_size:int, num_classes:int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # embedded: (batch_size, seq_len, emb_size)
        embedded = self.embedding(x)
        _, (h_n, _) = self.lstm(embedded)  # h_n: (1, batch_size, hidden_size)
        return self.fc(h_n.squeeze(0))  # (batch_size, num_classes)
