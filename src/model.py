import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class DoubleConv(nn.Sequential):
    def __init__(self, input_channels, output_channels):
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True),
        )


class UpBlock(nn.Module):
    def __init__(self, input_channels, skip_channels, output_channels):
        super().__init__()
        self.convolutions = DoubleConv(input_channels + skip_channels, output_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.convolutions(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.stage1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.stage2 = backbone.layer2
        self.stage3 = backbone.layer3
        self.stage4 = backbone.layer4
        self.input_features = DoubleConv(3, 16)
        self.up4 = UpBlock(512, 256, 128)
        self.up3 = UpBlock(128, 128, 64)
        self.up2 = UpBlock(64, 64, 32)
        self.up1 = UpBlock(32, 64, 16)
        self.up0 = UpBlock(16, 16, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def encoder_modules(self):
        return [self.stem, self.stage1, self.stage2, self.stage3, self.stage4]

    def set_encoder_trainable(self, trainable):
        for module in self.encoder_modules():
            for parameter in module.parameters():
                parameter.requires_grad_(trainable)
            if not trainable:
                module.eval()

    def forward(self, images):
        full = self.input_features(images)
        x0 = self.stem(images)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x = self.up4(x4, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x0)
        return self.head(self.up0(x, full))
