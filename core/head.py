import torch
import torch.nn as nn


class DropConnect(nn.Module):
    def __init__(self, drop_rate):
        super(DropConnect, self).__init__()
        self.p = drop_rate

    def forward(self, inputs):
        """Drop connect.
            Args:
                input (tensor: BCWH): Input of this structure.
                p (float: 0.0~1.0): Probability of drop connection.
                training (bool): The running mode.
            Returns:
                output: Output after drop connection.
        """
        p = self.p
        assert 0 <= p <= 1, 'p must be in range of [0,1]'

        if not self.training:
            return inputs

        batch_size = inputs.shape[0]
        keep_prob = 1 - p

        # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
        random_tensor = keep_prob
        random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
        binary_tensor = torch.floor(random_tensor)

        output = inputs / keep_prob * binary_tensor
        return output


def get_detector(name, **kwargs):
    if 'convs' == name:
        return Conv3x3ReLUBNs(kwargs['in_channels'],
                              kwargs['inner_channels'],
                              kwargs['out_channels'],
                              kwargs['scale'],
                              kwargs['num_convs'],
                              kwargs.get('drop_rate', 0.)
                              )
    raise ValueError(f'{name} is not supported.')


def Conv3x3ReLUBNs(in_channels,
                   inner_channels,
                   out_channels,
                   scale,
                   num_convs,
                   drop_rate=0.):
    layers = [nn.Sequential(
        nn.Conv2d(in_channels, inner_channels, 3, 1, 1),
        nn.ReLU(True),
        nn.BatchNorm2d(inner_channels),
        DropConnect(drop_rate) if drop_rate > 0. else nn.Identity()
    )]
    layers += [nn.Sequential(
        nn.Conv2d(inner_channels, inner_channels, 3, 1, 1),
        nn.ReLU(True),
        nn.BatchNorm2d(inner_channels),
        DropConnect(drop_rate) if drop_rate > 0. else nn.Identity()
    ) for _ in range(num_convs - 1)]

    cls_layer = nn.Conv2d(inner_channels, out_channels, 3, 1, 1)
    layers.append(cls_layer)
    layers.append(nn.UpsamplingBilinear2d(scale_factor=scale))
    return nn.Sequential(*layers)
