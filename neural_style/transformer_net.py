import torch
import torch.nn as nn

class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = nn.GroupNorm(32, 32)   # ✅ ONNX friendly

        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = nn.GroupNorm(64, 64)

        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = nn.GroupNorm(128, 128)

        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.GroupNorm(64, 64)

        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.GroupNorm(32, 32)

        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)

        self.relu = nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))

        # Residual blocks
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)

        # Upsample
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(nn.Module):
    """ResidualBlock"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.GroupNorm(channels, channels)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.GroupNorm(channels, channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + x
        return out


class UpsampleConvLayer(nn.Module):
    """UpsampleConvLayer using ConvTranspose2d"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        padding = kernel_size // 2
        output_padding = upsample - 1
        self.conv2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=upsample,
            padding=padding,
            output_padding=output_padding
        )

    def forward(self, x):
        out = self.conv2d(x)
        return out
