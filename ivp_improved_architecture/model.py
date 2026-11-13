"""Shared image encoder and order-aware temporal attention for action recognition."""
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class ViewInvariantActionNet(nn.Module):
    """Classify a single camera clip; paired cameras share all model weights."""
    def __init__(self, num_classes=10, frames=8, hidden=256, pretrained=True, dropout=0.2):
        super().__init__()
        if hidden % 4: raise ValueError('hidden must be divisible by four')
        self.frames = frames
        encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        encoder.fc = nn.Identity()
        self.encoder = encoder
        self.project = nn.Sequential(nn.Linear(512, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.position = nn.Parameter(torch.zeros(1, frames, hidden))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=hidden*2,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.attention = nn.Linear(hidden, 1)
        self.norm = nn.LayerNorm(hidden)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def forward(self, clips):
        if clips.ndim != 5 or clips.shape[1] != self.frames:
            raise ValueError(f'Expected [batch,{self.frames},3,height,width], got {tuple(clips.shape)}')
        batch, steps, channels, height, width = clips.shape
        if channels != 3: raise ValueError('Expected RGB input')
        features = self.encoder(clips.reshape(batch*steps, channels, height, width))
        tokens = self.project(features).reshape(batch, steps, -1) + self.position
        tokens = self.temporal(tokens)
        weights = self.attention(tokens).softmax(dim=1)
        embedding = self.norm((tokens * weights).sum(dim=1))
        return self.classifier(embedding), embedding

def paired_loss(logits, embeddings, labels, consistency_weight=0.1):
    """Cross entropy on both views plus cosine distance for paired events."""
    if logits.ndim != 3 or logits.shape[1] != 2:
        raise ValueError('Expected logits with shape [batch,2,classes]')
    targets = labels[:, None].expand(-1, 2).reshape(-1)
    classification = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets, label_smoothing=0.05)
    alignment = (1 - F.cosine_similarity(embeddings[:, 0], embeddings[:, 1], dim=-1)).mean()
    return classification + consistency_weight * alignment, classification.detach(), alignment.detach()
