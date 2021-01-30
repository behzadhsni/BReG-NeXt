
import torch
import itertools

class BRegNextShortcutModifier(torch.nn.Model):

    def __init__(self,):
        self._a = torch.nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self._c = torch.nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, inputs):
        numl = torch.atan((self._a * inputs) / torch.sqrt(self._c ** 2 + 1))
        denom = self._a * torch.sqrt(self._c ** 2 + 1)
        return  (numl / denom)


class BReGNeXtResidualLayer(torch.nn.Model):

    def __init__(self, in_channels: int, out_channels: int, downsample_stride: int=1):
        self._out_channels = out_channels
        self._in_channels = in_channels
        self._downsample_stride = 1 if downsample is None else downsample

        # TODO: These may have slightly different layer initializations.
        self._conv0 = torch.nn.Conv2d(in_channels, out_channels, 3, downsample_stride)
        self._conv0 = torch.nn.Conv2d(out_channels, out_channels, 3, 1)
        self._shortcut = BRegNextShortcutModifier()
        self._batchnorm_conv0 = torch.nn.BatchNorm2d(self._in_channels)
        self._batchnorm_conv1 = torch.nn.BatchNorm2d(self._out_channels)

    def forward(self, inputs):
        # First convolution
        normed_inputs = inputs if self._batchnorm_conv0 is None else self._batchnorm_conv0(inputs)
        normed_inputs = torch.nn.functional.elu(normed_inputs)
        conv0_outputs = self._conv0(normed_inputs)

        # Second convolution
        normed_conv0_outputs = conv0_outputs if self._batchnorm_conv1 is None else self._batchnorm_conv1(conv0_outputs)
        normed_conv0_outputs = torch.nn.functional.elu(normed_conv0_outputs)
        conv1_outputs = self._conv1(normed_conv0_outputs)

        # TODO: Are the shortcut weights always the same for every layer?
        shortcut_modifier = self._shortcut(inputs)
        if self._downsample_stride > 1:
            shortcut_modifier = torch.nn.functional.avg_pool2d(shortcut_modifier, self._downsample_stride, self._downsample_stride)

        # Downsample the projection
        if self._out_channels != self._in_channels:
            pad_dimension = (self._out_channels - self._in_channels) // 2
            shortcut_modifier = torch.nn.functional.pad(shortcut_modifier, [pad_dimension, pad_dimension])

        return conv1_outputs + shortcut_modifier


class BRegNextResidualBlock(torch.nn.Module):

    def forward(self, inputs):
        # TODO: Adjust the layer stack, since the number of channels will be different for the first block vs. the other blocks
        return self._layer_stack(inputs)


class BReG_NeXt(torch.nn.Module):

    def __init__(self, n_classes: int = 8) -> None:
        self._model = torch.nn.Sequential(
            torch.nn.Conv2D(in_channels=3, out_channels=32),
            BRegNextResidualBlock(n_blocks=7, in_channels=32, out_channels=32),
            BRegNextResidualBlock(n_blocks=1, in_channels=32, out_channels=64, downsample_stride=2),
            BRegNextResidualBlock(n_blocks=8, in_channels=64, out_channels=64),
            BRegNextResidualBlock(n_blocks=1, in_channels=64, out_channels=128, downsample_stride=2),
            BRegNextResidualBlock(n_blocks=7, in_channels=128, out_channels=128),
            torch.nn.BatchNorm2d(128),
            torch.nn.ELU(),
            torch.nn.AdaptiveAvgPool2d((1,1)),
        )
        self._fc0 = torch.nn.Linear(128, n_classes)

    def conv0_params(self,):
        return self._conv0.parameters()

    def model_params(self,):
        return itertools.chain.from_iterable([self._fc0.parameters(), self._model.parameters()])

    def forward(self, x):
        return self._fc0(self._model(x).reshape(-1, 128))
