# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.embeddings import ScalarEmbedding
from layers.embeddings import CategoricalEmbedding
from layers.convolutional import SimpleConv
from layers.recurrent import SimpleGRU
from layers.attention import VaswaniAttention
from features.custom_features import SPATIAL_FEATURES


__all__ = ['SimpleConvLSTM']


class SimpleConvLSTM(nn.Module):
    """Add class docstring."""
    def __init__(self,
                 embedding_dim,
                 rnn_input_size,
                 rnn_hidden_size,
                 output_size,
                 include=['height_map', 'visibility_map', 'player_relative', 'unit_type']):
        super(SimpleConvLSTM, self).__init__()

        self.embedding_dim = embedding_dim
        self.rnn_input_size = rnn_input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.output_size = output_size
        self.include = include

        self.embedding_dims = {
            'height_map': 10,
            'visibility_map': 10,
            'player_relative': 10,
            'unit_type': 100
        }

        self.cnn_channel_size = 0

        self.feat_names = [k for k in SPATIAL_FEATURES._asdict()]  # redundant?

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

        # Convolution module.
        self.conv = SimpleConv(
            in_channels=self.cnn_channel_size,
            output_size=self.rnn_input_size,
        )

        # Recurrent module.
        self.gru = SimpleGRU(
            input_size=self.rnn_input_size,
            hidden_size=self.rnn_hidden_size,
            output_size=self.output_size,
        )

        # Attention module.
        self.attn = VaswaniAttention(
            hidden_size=self.rnn_hidden_size,
            context_size=self.rnn_hidden_size,
        )

        self.linear = nn.Linear(self.rnn_hidden_size, self.output_size)

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
        embedded = embedded.permute(1, 0, 2, 3, 4)  # (T, B, cnn_channel_size, H, W)

        conv_outputs = []
        for emb in embedded:
            conv_out = self.conv(emb)                      # (B, rnn_input_size)
            conv_outputs.append(conv_out)
        conv_outputs = torch.stack(conv_outputs, dim=1)    # (B, T, rnn_input_size)

        encoder_outputs, hidden = self.gru(conv_outputs)   # (B, T, hidden_size), (B, hidden_size)
        attn_probs = self.attn(encoder_outputs, hidden)    # (B, T)
        attn_probs = attn_probs.unsqueeze(1)               # (B, 1, T)

        weighted = torch.bmm(attn_probs, encoder_outputs)  # (B, 1, hidden_size)
        weighted = weighted.squeeze(1)                     # (B, hidden_size)
        weighted = F.leaky_relu(weighted)

        logits = self.linear(weighted)

        return logits
