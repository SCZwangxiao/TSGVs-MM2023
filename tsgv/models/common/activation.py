import torch.nn as nn
import torch.nn.functional as F

def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

def get_activation_nn(activation):
    if activation is None:
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))