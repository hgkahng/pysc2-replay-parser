# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.convolutional import BasicBlock3D
from layers.embeddings import ScalarEmbedding
from layers.embeddings import CategoricalEmbedding
from layers.convolutional import SimpleConv
from layers.recurrent import SimpleGRU
from layers.attention import VaswaniAttention
from features.custom_features import SPATIAL_FEATURES


class ResNet3D(nn.Module):
    """Add class docstring."""
    def __init__(self, embedding_dims=None, num_classes=2, include=['unit_type', 'player_relative']):
        super(ResNet3D, self).__init__()

        if embedding_dims is None:
            self.embedding_dims = {
                'height_map': 10,
                'visibility_map': 10,
                'player_relative': 10,
                'unit_type': 100
            }
        else:
            assert isinstance(embedding_dims, dict)
            self.embedding_dims = embedding_dims

        self.num_classes = num_classes
        self.include = include
        self.cnn_channel_size = 0

        """Embedding layers."""
        self.embeddings = nn.ModuleDict()
        for name, feat in SPATIAL_FEATURES._asdict().items():
            if name not in self.include:
                continue
            feat_type = str(feat.type).split('.')[-1]
            if feat_type == 'CATEGORICAL':
                self.embeddings[name] = CategoricalEmbedding(
                    category_size=feat.scale,
                    embedding_dim=self.embedding_dims.get(name),
                    name=name,
                )
            elif feat_type == 'SCALAR':
                self.embeddings[name] = ScalarEmbedding(
                    embedding_dim=self.embedding_dims.get(name),
                    name=name
                )
            else:
                raise NotImplementedError
            self.cnn_channel_size += self.embedding_dims.get(name)

        self.conv1 = nn.Conv3d(self.cnn_channel_size, 32, 7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.block1 = BasicBlock3D(32, 64, stride=1, downsample=None)
        self.pool1 = nn.MaxPool3d(kernel_size=(5, 4, 4), padding=0)
        self.block2 = BasicBlock3D(64, 128, stride=2, downsample=None)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), padding=0)
        self.block3 = BasicBlock3D(128, 256, stride=2, downsample=None)
        self.pool3 = nn.AvgPool3d(kernel_size=(5, 4, 4), stride=1, padding=0)

        self.linear = nn.Linear(256, self.num_classes)

    def forward(self, inputs):
        """
        Arguments:
            inputs: dict, with feature name as keys and 4d tensors as values.
                each 4d tensor has shape (B, T, H, W). A list of 3D tensors with shape (T, H, W) is
                also allowed.
        """
        assert isinstance(inputs, dict)

        embedded = []
        for name, x in inputs.items():
            emb_out = self.embeddings[name](x)
            embedded.append(emb_out.float())
        embedded = torch.cat(embedded, dim=2)       # (B, T, cnn_channel_size, H, W)
        embedded = embedded.permute(0, 2, 1, 3, 4)  # (B, cnn_channel_size, T, H, W)

        x = self.conv1(embedded)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool3(x)

        logits = self.linear(x.view(x.size(0), -1))

        return logits
