import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    densenet121, DenseNet121_Weights,
    densenet169, DenseNet169_Weights,
    densenet201, DenseNet201_Weights,
)


def _make_regressor(in_features: int, dropout: float = 0.3) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(256, 1),
    )


def _init_weights(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class _WheatDenseNet(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int,
                 in_channels: int, dropout: float):
        super().__init__()

        backbone.features.conv0 = nn.Conv2d(
            in_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )

        self.features   = backbone.features
        self.regressor  = _make_regressor(feat_dim, dropout)

        _init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        return self.regressor(x).squeeze(1)


class WheatDenseNet121(_WheatDenseNet):
    def __init__(self, in_channels: int, dropout: float = 0.3, **kwargs):
        bb = densenet121(weights=None)
        super().__init__(bb, feat_dim=1024,
                         in_channels=in_channels, dropout=dropout)


class WheatDenseNet169(_WheatDenseNet):
    def __init__(self, in_channels: int, dropout: float = 0.3, **kwargs):
        bb = densenet169(weights=None)
        super().__init__(bb, feat_dim=1664,
                         in_channels=in_channels, dropout=dropout)


class WheatDenseNet201(_WheatDenseNet):
    def __init__(self, in_channels: int, dropout: float = 0.3, **kwargs):
        bb = densenet201(weights=None)
        super().__init__(bb, feat_dim=1920,
                         in_channels=in_channels, dropout=dropout)


DENSENET_REGISTRY = {
    "densenet121": WheatDenseNet121,
    "densenet169": WheatDenseNet169,
    "densenet201": WheatDenseNet201,
}

DENSENET_FEAT_DIMS = {
    "densenet121": 1024,
    "densenet169": 1664,
    "densenet201": 1920,
}


def build_densenet(name: str, in_channels: int,
                   dropout: float = 0.3) -> nn.Module:
    assert name in DENSENET_REGISTRY, \
        (f"[densenet_model.py] Unknown model '{name}'. "
         f"Available: {list(DENSENET_REGISTRY.keys())}")
    return DENSENET_REGISTRY[name](in_channels=in_channels, dropout=dropout)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("densenet_model.py self-check (all trained from scratch)")
    print(f"Device: {device}")
    print("=" * 65)

    dummy = torch.randn(4, 10, 64, 64).to(device)

    for name in DENSENET_REGISTRY.keys():
        try:
            model = build_densenet(name, in_channels=10).to(device)
            model.eval()
            with torch.no_grad():
                out = model(dummy)
            params   = count_parameters(model)
            expected_feat = DENSENET_FEAT_DIMS[name]
            status   = "OK" if out.shape == (4,) else "FAIL"
            reg_head = str(model.regressor).replace("\n", " ")

            print(f"  {status} {name:<15} | output={tuple(out.shape)} | "
                  f"params={params/1e6:.1f}M | feat_dim={expected_feat} | "
                  f"out_range=[{out.min():.2f}, {out.max():.2f}]")
            print(f"           regressor: {reg_head}")
        except Exception as e:
            print(f"  FAIL {name:<15} | ERROR: {e}")

    print()
    print("Run run_densenet_compare.py to start the experiment after self-check passes.")
    print()
    print("Note: parameter counts of the three models compared with ResNet ablation baseline:")
    print("  densenet121  ~  8.0M  vs  resnet18  ~ 11.2M  (lighter)")
    print("  densenet169  ~ 14.3M  vs  resnet50  ~ 24.1M  (in between)")
    print("  densenet201  ~ 20.1M  vs  resnet50  ~ 24.1M  (closest)")