import torch
import torch.nn as nn
from torchtext.vocab import GloVe


class TextCNN(nn.Module):
    def __init__(self, vocab, embed_size=100, kernel_sizes=[3, 4, 5], num_channels=[100] * 3):
        super().__init__()
        self.glove = GloVe(name="6B", dim=100)
        self.unfrozen_embedding = nn.Embedding.from_pretrained(self.glove.get_vecs_by_tokens(vocab.get_itos()), padding_idx=vocab['<pad>'])
        self.frozen_embedding = nn.Embedding.from_pretrained(self.glove.get_vecs_by_tokens(vocab.get_itos()),
                                                             padding_idx=vocab['<pad>'],
                                                             freeze=True)

        self.convs_for_unfrozen = nn.ModuleList()
        self.convs_for_frozen = nn.ModuleList()
        for out_channels, kernel_size in zip(num_channels, kernel_sizes):
            self.convs_for_unfrozen.append(nn.Conv1d(in_channels=embed_size, out_channels=out_channels, kernel_size=kernel_size))
            self.convs_for_frozen.append(nn.Conv1d(in_channels=embed_size, out_channels=out_channels, kernel_size=kernel_size))

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.relu = nn.ReLU()
        # dropout，提升模型泛化能力
        self.dropout = nn.Dropout(0.5)
        # 逻辑回归做二分类
        self.fc = nn.Linear(sum(num_channels) * 2, 2)
        self.apply(self._init_weights)

    def forward(self, x):
        x_unfrozen = self.unfrozen_embedding(x).transpose(1, 2)  # (batch_size, embed_size, seq_len)
        x_frozen = self.frozen_embedding(x).transpose(1, 2)  # (batch_size, embed_size, seq_len)
        # 池化后得到的向量
        pooled_vector_for_unfrozen = [self.pool(self.relu(conv(x_unfrozen))).squeeze()
                                      for conv in self.convs_for_unfrozen]  # shape of each element: (batch_size, 100)
        pooled_vector_for_frozen = [self.pool(self.relu(conv(x_frozen))).squeeze()
                                    for conv in self.convs_for_frozen]  # shape of each element: (batch_size, 100)
        # 将向量拼接起来后得到一个更长的向量
        feature_vector = torch.cat(pooled_vector_for_unfrozen + pooled_vector_for_frozen, dim=-1)  # (batch_size, 600)
        output = self.fc(self.dropout(feature_vector))  # (batch_size, 2)
        return output

    def _init_weights(self, m):
        # 仅对线性层和卷积层进行xavier初始化
        if type(m) in (nn.Linear, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)