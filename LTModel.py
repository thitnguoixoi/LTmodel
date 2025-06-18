import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import json
import logging

# Cấu hình logger
logger = logging.getLogger(__name__)


class TokenEmbedding(nn.Module):
    """
    Lớp embedding token chuyển đổi token id thành vector nhúng
    """

    def __init__(self, vocab_size, hidden_size):
        """
        Khởi tạo lớp embedding

        Args:
            vocab_size: Kích thước từ điển
            hidden_size: Kích thước vector nhúng
        """
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.hidden_size = hidden_size

        # Khởi tạo trọng số
        self.reset_parameters()

    def reset_parameters(self):
        """Khởi tạo trọng số embedding"""
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

    def forward(self, x):
        """
        Forward pass qua lớp embedding

        Args:
            x: Tensor chứa token ids [batch_size, seq_length]

        Returns:
            Tensor chứa vectors nhúng [batch_size, seq_length, hidden_size]
        """
        return self.embedding(x) * math.sqrt(self.hidden_size)


class PositionalEncoding(nn.Module):
    """
    Lớp mã hóa vị trí (positional encoding) cho transformer
    """

    def __init__(self, hidden_size, max_position=512, dropout=0.1):
        """
        Khởi tạo positional encoding

        Args:
            hidden_size: Kích thước của các vectors nhúng
            max_position: Độ dài tối đa của chuỗi
            dropout: Tỷ lệ dropout áp dụng
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Tạo ma trận positional encoding
        # Shape: [max_position, hidden_size]
        pe = torch.zeros(max_position, hidden_size)
        # Tạo tensor chứa các vị trí
        position = torch.arange(
            0, max_position, dtype=torch.float).unsqueeze(1)
        # Tạo tensor chứa các divisors
        div_term = torch.exp(torch.arange(
            0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))

        # Tính toán positional encoding
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Thêm batch dimension và đăng ký buffer
        pe = pe.unsqueeze(0)  # [1, max_position, hidden_size]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Thêm positional encoding vào embedding

        Args:
            x: Tensor đầu vào [batch_size, seq_length, hidden_size]

        Returns:
            Tensor với positional encoding [batch_size, seq_length, hidden_size]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MyfAttention(nn.Module):
    """
    Multi-head attention
    """

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        """
        Khởi tạo lớp multi-head attention

        Args:
            hidden_size: Kích thước của vectors nhúng
            num_heads: Số lượng attention heads
            dropout: Tỷ lệ dropout áp dụng
        """
        super(MyfAttention, self).__init__()
        self.head_size = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.scaling = self.head_size ** -0.5
        # Replace random initialization with proper scaling
        self.W_Q = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.02)
        self.W_K = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.02)
        self.W_V = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.02)
        self.W_O = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.02)

        self.dropout = nn.Dropout(dropout)

    def forward(self, X, mask=None):
        """
        Forward pass của multi-head attention

        Args:
            X: Tensor đầu vào [batch_size, seq_length, hidden_size]
            mask: Optional mask tensor [batch_size, seq_length, seq_length]

        Returns:
            Tensor đầu ra [batch_size, seq_length, hidden_size]
        """
        batch_size, seq_len, _ = X.size()

        # Tính Q, K, V từ X
        Q = torch.matmul(X, self.W_Q)  # [batch_size, seq_len, hidden_size]
        K = torch.matmul(X, self.W_K)
        V = torch.matmul(X, self.W_V)

        # Chia thành các đầu attention
        Q = Q.view(batch_size, seq_len, self.num_heads,
                   self.head_size).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads,
                   self.head_size).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads,
                   self.head_size).transpose(1, 2)
        # Q, K, V: [batch_size, num_heads, seq_len, head_size]

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling

        scores = torch.clamp(scores, min=-1e4, max=1e4)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(
                    2)  # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size)

        output = torch.matmul(context, self.W_O)

        return output


class FeedForward(nn.Module):
    """
    Optimized Feed-Forward neural network with GELU activation
    """

    def __init__(self, hidden_size, ff_size, dropout=0.1):
        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(hidden_size, ff_size)
        self.linear2 = nn.Linear(ff_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize feed-forward weights"""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.linear2(x)


class TLayer(nn.Module):
    """
    Một layer của transformer encoder
    """

    def __init__(self, hidden_size, num_heads, ff_size, dropout=0.1):
        """
        Khởi tạo transformer layer

        Args:
            hidden_size: Kích thước của vectors nhúng
            num_heads: Số lượng attention heads
            ff_size: Kích thước hidden layer trong feed-forward
            dropout: Tỷ lệ dropout áp dụng
        """
        super(TLayer, self).__init__()

        # Multi-head attention
        self.attention = MyfAttention(hidden_size, num_heads, dropout)

        # Feed-forward network
        self.feed_forward = FeedForward(hidden_size, ff_size, dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass của transformer layer

        Args:
            x: Tensor đầu vào [batch_size, seq_length, hidden_size]
            mask: Optional mask tensor [batch_size, seq_length, seq_length]

        Returns:
            Tensor đầu ra [batch_size, seq_length, hidden_size]
        """
        # Multi-head attention với residual connection
        norm_x = self.norm1(x)
        # Fix: Pass only norm_x and mask
        attention_output = self.attention(norm_x, mask)
        x = x + self.dropout(attention_output)

        # Feed-forward network với residual connection
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout(ff_output)

        return x


class TEncoder(nn.Module):
    """
    Encoder của transformer
    """

    def __init__(self, num_layers, hidden_size, num_heads, ff_size, dropout=0.1):
        """
        Khởi tạo transformer encoder

        Args:
            num_layers: Số lượng transformer layers
            hidden_size: Kích thước của vectors nhúng
            num_heads: Số lượng attention heads
            ff_size: Kích thước hidden layer trong feed-forward
            dropout: Tỷ lệ dropout áp dụng
        """
        super(TEncoder, self).__init__()

        # Stack các transformer layers
        self.layers = nn.ModuleList([
            TLayer(hidden_size, num_heads, ff_size, dropout)
            for _ in range(num_layers)
        ])

        # Layer normalization cuối cùng
        self.norm = nn.LayerNorm(hidden_size)

        # Add gradient checkpointing flag
        # self.gradient_checkpointing = False

    def forward(self, x, mask=None):
        outputs = []
        for layer in self.layers:
            outputs.append(layer(x, mask))  # Mỗi layer xử lý độc lập

        # Tổng hợp kết quả từ các layer (ví dụ: cộng hoặc trung bình)
        x = torch.stack(outputs, dim=0).mean(
            dim=0)  # Trung bình kết quả từ các layer

        # Layer normalization cuối cùng
        x = self.norm(x)

        return x


class LTModel(nn.Module):
    """
    Mô hình transformer cho phân tích và phân loại dữ liệu HTTP
    """

    def __init__(self, vocab_size, hidden_size=512, num_layers=2, num_heads=8,
                 ff_size=2048, max_position=512, dropout=0.1, num_classes=2):
        """
        Khởi tạo LT Model

        Args:
            vocab_size: Kích thước vocabulary
            hidden_size: Kích thước của vectors nhúng
            num_layers: Số lượng transformer layers
            num_heads: Số lượng attention heads
            ff_size: Kích thước hidden layer trong feed-forward
            max_position: Độ dài tối đa của chuỗi
            dropout: Tỷ lệ dropout áp dụng
            num_classes: Số lượng lớp đầu ra cho phân loại
        """
        super(LTModel, self).__init__()

        # Add model configuration
        self.config = {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'ff_size': ff_size,
            'max_position': max_position,
            'dropout': dropout,
            'num_classes': num_classes
        }

        # Enable mixed precision training
        self.scaler = torch.amp.GradScaler('cuda')

        # Add gradient checkpointing flag as class attribute
        self.gradient_checkpointing = False

        # Initialize components
        self._init_components()

    def _init_components(self):
        # Initialize all components with optimized settings
        self.token_embedding = TokenEmbedding(self.config['vocab_size'],
                                              self.config['hidden_size'])
        self.position_encoding = PositionalEncoding(self.config['hidden_size'],
                                                    self.config['max_position'],
                                                    self.config['dropout'])
        self.encoder = TEncoder(self.config['num_layers'],
                                self.config['hidden_size'],
                                self.config['num_heads'],
                                self.config['ff_size'],
                                self.config['dropout'])
        self.pre_classifier = nn.LayerNorm(self.config['hidden_size'])
        self.classifier = nn.Linear(self.config['hidden_size'],
                                    self.config['num_classes'])

        # Khởi tạo trọng số classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def create_attention_mask(self, input_ids):
        batch_size, seq_length = input_ids.size()
        attention_mask = (input_ids != 0).float()
        # Reshape mask to [batch_size, 1, 1, seq_length] for broadcasting
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        return attention_mask

    def forward(self, input_ids, attention_mask=None, labels=None):
        if self.training and self.gradient_checkpointing:
            self.encoder.gradient_checkpointing = True
        else:
            self.encoder.gradient_checkpointing = False

        if attention_mask is None:
            attention_mask = self.create_attention_mask(input_ids)

        # Ensure attention mask has correct shape
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        embeddings = self.token_embedding(input_ids)
        embeddings = self.position_encoding(embeddings)

        sequence_output = self.encoder(embeddings, attention_mask)

        # Efficient classification
        cls_output = sequence_output[:, 0, :]
        cls_output = self.pre_classifier(cls_output)
        logits = self.classifier(cls_output)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config['num_classes']),
                labels.view(-1),
                label_smoothing=0.1  # Add label smoothing for numerical stability
            )

        return {'logits': logits, 'loss': loss}
