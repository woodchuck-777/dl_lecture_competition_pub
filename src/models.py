import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.multi_head_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.multi_head_attn(x, x, x)
        return self.layer_norm(attn_output + x)


class AttentionEEGClassifier(nn.Module):
    def __init__(self, num_classes, num_channels, embed_dim=64, num_heads=4):
        super(AttentionEEGClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(num_channels, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.flatten = nn.Flatten(start_dim=2)
        self.self_attention = SelfAttention(embed_dim, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(281, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = x.permute(2, 0, 1)
        x = self.self_attention(x)
        x = x.permute(1, 2, 0)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x
