# -*- coding: utf-8 -*-

import torch.nn as nn

def count_parameters(model):
    assert isinstance(model, nn.Module)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
