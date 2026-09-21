import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class WheatResNet50(nn.Module):
    def __init__(
        self,
        in_channels: int,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels

        if pretrained:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            backbone = resnet50(weights=None)

        orig_conv1 = backbone.conv1
        new_conv1 = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = 64,
            kernel_size  = orig_conv1.kernel_size,
            stride       = orig_conv1.stride,
            padding      = orig_conv1.padding,
            bias         = orig_conv1.bias is not None,
        )

        if pretrained:
            orig_weight = orig_conv1.weight.data
            mean_weight = orig_weight.mean(dim=1, keepdim=True)
            new_weight  = mean_weight.repeat(1, in_channels, 1, 1)
            new_conv1.weight.data = new_weight

        backbone.conv1 = new_conv1

        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        self.layer4  = backbone.layer4
        self.avgpool = backbone.avgpool

        self.regressor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)

        x = self.regressor(x)
        return x.squeeze(1)


def build_model(in_channels: int, pretrained: bool = True, dropout: float = 0.3) -> WheatResNet50:
    return WheatResNet50(in_channels=in_channels, pretrained=pretrained, dropout=dropout)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import os
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 55)
    print("model.py self-check")
    print(f"Device: {device}")
    print("=" * 55)

    test_cases = [
        ("G0", 7),
        ("G1", 9),
        ("G2", 9),
        ("G3", 10),
        ("G4", 10),
        ("G5", 10),
        ("G6", 13),
    ]

    for g_name, c in test_cases:
        model = build_model(in_channels=c, pretrained=True).to(device)
        model.eval()

        dummy_input = torch.randn(4, c, 64, 64).to(device)
        with torch.no_grad():
            output = model(dummy_input)

        params  = count_parameters(model)
        status  = "OK" if output.shape == (4,) else "FAIL"
        print(f"  {status} {g_name} (C={c:2d}): "
              f"input={tuple(dummy_input.shape)} -> "
              f"output={tuple(output.shape)}, "
              f"params={params/1e6:.1f}M, "
              f"output_range=[{output.min():.2f}, {output.max():.2f}]")

    print("\nSelf-check complete, model.py is ready to use.")