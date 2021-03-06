import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FC(nn.Module):
    """
    3-layer fully connected network
    """
    def __init__(self, input_dim, output_dim):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        out = self.fc(x)
        return out

class Duel_FC(nn.Module):
    """
    3-layer fully connected network for dueling architecture
    """
    def __init__(self, input_dim, output_dim):
        super(Duel_FC, self).__init__()
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.adv_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        value = self.value_net(x)
        adv = self.adv_net(x)
        out = value + (adv - adv.mean(dim=1, keepdims=True))
        return out

class Conv_2D(nn.Module):
    """
    4-layer 2D convolutional neural network
    """
    def __init__(self, input_shape, output_dim):
        super(Conv_2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2),
            nn.ReLU(),
            nn.Conv1d(8, 16, 3, stride=2),
            nn.ReLU(),
        )
        conv_out_dim =   self._get_conv_out_dim(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def _get_conv_out_dim(self, input_shape):
        out = self.conv(torch.zeros(1, *input_shape))
        return int(np.prod(out.shape))

    def forward(self, x):
        batch_size = x.shape[0]
        conv_out = self.conv(x).view(batch_size,-1)
        out = self.fc(conv_out)
        return out

class Duel_Conv_2D(nn.Module):
    """
    4-layer two-branch 1D convolutional neural network for dueling architecture,
    the first two 1D convolutional layer are shared.
    """
    def __init__(self, input_shape, output_dim):
        super(Duel_Conv_2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(8,16, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16,32, 3, stride=2),
            nn.ReLU(),
        )
        conv_out_dim =  self._get_conv_out_dim(input_shape)
        self.value_net = nn.Sequential(
            nn.Linear(conv_out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.adv_net = nn.Sequential(
            nn.Linear(conv_out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def _get_conv_out_dim(self, input_shape):
        out = self.conv(torch.zeros(1, *input_shape))
        return int(np.prod(out.shape))

    def forward(self, x):
        batch_size = x.shape[0]
        conv_out = self.conv(x).view(batch_size,-1)
        value = self.value_net(conv_out)
        adv = self.adv_net(conv_out)
        out = value + (adv - adv.mean(dim=1, keepdims=True))
        return out