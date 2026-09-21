import torch
import torch.nn as nn
from torchvision.models import (
    resnet50,
    densenet121,
    efficientnet_b0,
    mobilenet_v3_small,
    convnext_tiny,
)
import timm


def _make_regressor(in_features: int, dropout: float = 0.3) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(256, 1),
    )


def _make_regressor_small(in_features: int, dropout: float = 0.3) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(128, 1),
    )


class WheatResNet50(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.3, **kwargs):
        super().__init__()
        bb = resnet50(weights=None)

        bb.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
                             stride=2, padding=3, bias=False)

        self.features = nn.Sequential(
            bb.conv1, bb.bn1, bb.relu, bb.maxpool,
            bb.layer1, bb.layer2, bb.layer3, bb.layer4,
        )
        self.pool      = nn.AdaptiveAvgPool2d(1)
        self.regressor = _make_regressor(2048, dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.regressor(x).squeeze(1)


class WheatDenseNet121(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.3, **kwargs):
        super().__init__()
        bb = densenet121(weights=None)

        bb.features.conv0 = nn.Conv2d(in_channels, 64, kernel_size=7,
                                      stride=2, padding=3, bias=False)

        self.features  = bb.features
        self.pool      = nn.AdaptiveAvgPool2d(1)
        self.regressor = _make_regressor(1024, dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = torch.relu(x)
        x = self.pool(x).flatten(1)
        return self.regressor(x).squeeze(1)


class WheatEfficientNetB0(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.3, **kwargs):
        super().__init__()
        bb = efficientnet_b0(weights=None)

        orig_conv = bb.features[0][0]
        bb.features[0][0] = nn.Conv2d(
            in_channels, orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False,
        )

        self.features  = bb.features
        self.pool      = nn.AdaptiveAvgPool2d(1)
        self.regressor = _make_regressor(1280, dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.regressor(x).squeeze(1)


class WheatMobileNetV3Small(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.3, **kwargs):
        super().__init__()
        bb = mobilenet_v3_small(weights=None)

        orig_conv = bb.features[0][0]
        bb.features[0][0] = nn.Conv2d(
            in_channels, orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False,
        )

        self.features  = bb.features
        self.pool      = nn.AdaptiveAvgPool2d(1)
        self.regressor = _make_regressor(576, dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.regressor(x).squeeze(1)


class WheatConvNeXtTiny(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.3, **kwargs):
        super().__init__()
        bb = convnext_tiny(weights=None)

        orig_conv = bb.features[0][0]
        bb.features[0][0] = nn.Conv2d(
            in_channels, orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=orig_conv.bias is not None,
        )

        self.features  = bb.features
        self.pool      = nn.AdaptiveAvgPool2d(1)
        self.regressor = _make_regressor(768, dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.regressor(x).squeeze(1)


class WheatEfficientViTB0(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.3, **kwargs):
        super().__init__()

        bb = timm.create_model(
            "efficientvit_b0",
            pretrained=False,
            in_chans=in_channels,
            num_classes=0,
        )

        self.backbone  = bb
        feat_dim       = bb.num_features
        self.regressor = _make_regressor_small(feat_dim, dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.backbone(x)
        return self.regressor(x).squeeze(1)


ARCH_REGISTRY = {
    "resnet50":        WheatResNet50,
    "densenet121":     WheatDenseNet121,
    "efficientnet_b0": WheatEfficientNetB0,
    "mobilenetv3_s":   WheatMobileNetV3Small,
    "convnext_tiny":   WheatConvNeXtTiny,
    "efficientvit_b0": WheatEfficientViTB0,
}


def build_arch(name: str, in_channels: int,
               pretrained: bool = False, dropout: float = 0.3) -> nn.Module:
    assert name in ARCH_REGISTRY, \
        f"[models.py] Unknown arch '{name}'. Available: {list(ARCH_REGISTRY.keys())}"
    return ARCH_REGISTRY[name](in_channels=in_channels, dropout=dropout)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("models.py v3 self-check (all trained from scratch)")
    print(f"Device: {device}")
    print("=" * 65)

    dummy = torch.randn(4, 10, 64, 64).to(device)

    for name in ARCH_REGISTRY.keys():
        try:
            model = build_arch(name, in_channels=10).to(device)
            model.eval()
            with torch.no_grad():
                out = model(dummy)
            params = count_parameters(model)
            status = "OK" if out.shape == (4,) else "FAIL"
            print(f"  {status} {name:<20} | output={tuple(out.shape)} | "
                  f"params={params/1e6:.1f}M | "
                  f"out_range=[{out.min():.2f}, {out.max():.2f}]")
        except Exception as e:
            print(f"  FAIL {name:<20} | ERROR: {e}")

    print("\nSelf-check complete.")