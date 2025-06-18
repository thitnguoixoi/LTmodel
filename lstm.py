import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import logging

# Cấu hình logger
logger = logging.getLogger(__name__)


class TokenEmbedding(nn.Module):
    """
    Lớp embedding token chuyển đổi token id thành vector nhúng
    """

    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.hidden_size = hidden_size
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.hidden_size)


class PositionalEncoding(nn.Module):
    """
    Lớp mã hóa vị trí (positional encoding) cho LSTM‑based encoder
    """

    def __init__(self, hidden_size, max_position=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_position, hidden_size)
        position = torch.arange(
            0, max_position, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() *
            (-math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # [1, max_position, hidden_size]
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class FeedForward(nn.Module):
    """Feed‑Forward network with GELU activation"""

    def __init__(self, hidden_size, ff_size, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, ff_size)
        self.linear2 = nn.Linear(ff_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.linear2(x)


class LSTMLayer(nn.Module):
    def __init__(self, hidden_size, ff_size, dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.feed_forward = FeedForward(hidden_size, ff_size, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x, mask=None
    ):  # mask kept for API compatibility, currently unused
        norm_x = self.norm1(x)
        rnn_out, _ = self.rnn(norm_x)
        x = x + self.dropout(rnn_out)

        norm_x = self.norm2(x)
        ff_out = self.feed_forward(norm_x)
        x = x + self.dropout(ff_out)
        return x


class LSTMEncoder(nn.Module):
    """Stack nhiều LSTMLayer"""

    def __init__(self, num_layers, hidden_size, ff_size, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [LSTMLayer(hidden_size, ff_size, dropout)
             for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        outputs = []
        for layer in self.layers:
            x = layer(x, mask)
            outputs.append(x)
        x = torch.stack(outputs, dim=0).mean(dim=0)
        return self.norm(x)


class ModelLSTM(nn.Module):
    """
    Mô hình sử dụng LSTM thay cho attention.
    """

    def __init__(
        self,
        vocab_size,
        hidden_size=512,
        num_layers=2,
        ff_size=2048,
        max_position=512,
        dropout=0.1,
        num_classes=2,
    ):
        super().__init__()
        self.config = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "ff_size": ff_size,
            "max_position": max_position,
            "dropout": dropout,
            "num_classes": num_classes,
        }
        self.token_embedding = TokenEmbedding(vocab_size, hidden_size)
        self.position_encoding = PositionalEncoding(
            hidden_size, max_position, dropout)
        self.encoder = LSTMEncoder(num_layers, hidden_size, ff_size, dropout)
        self.pre_classifier = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, labels=None):
        embeddings = self.token_embedding(input_ids)
        embeddings = self.position_encoding(embeddings)
        sequence_output = self.encoder(embeddings)
        cls_output = sequence_output[:, 0, :]
        cls_output = self.pre_classifier(cls_output)
        logits = self.classifier(cls_output)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config["num_classes"]),
                labels.view(-1),
                label_smoothing=0.1,
            )
        return {"logits": logits, "loss": loss}

    # Các hàm save_model và load_model được giữ nguyên cấu trúc nhưng
    # cập nhật tham số tương ứng với mô hình mới.
    def save_model(self, filepath):
        torch.save(self.state_dict(), f"{filepath}/model.pt")
        with open(f"{filepath}/config.json", "w") as f:
            json.dump(self.config, f, indent=2)

    @classmethod
    def load_model(cls, filepath, device="cpu"):
        with open(f"{filepath}/config.json", "r") as f:
            config = json.load(f)
        model = cls(**config)
        state_dict = torch.load(f"{filepath}/model.pt", map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        return model
