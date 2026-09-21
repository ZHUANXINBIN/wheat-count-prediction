import os
import sys
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import build_arch, count_parameters, ARCH_REGISTRY


def measure_one_arch(model: torch.nn.Module,
                     in_channels: int,
                     device: torch.device,
                     batch_size: int = 32,
                     img_size: int = 64,
                     warmup_runs: int = 10,
                     measure_runs: int = 100) -> dict:
    model = model.to(device)
    model.eval()

    dummy = torch.randn(batch_size, in_channels, img_size, img_size, device=device)

    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy)
    torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(measure_runs):
            starter.record()
            _ = model(dummy)
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))

    times = np.array(times)
    ms_per_batch  = float(np.mean(times))
    ms_per_sample = ms_per_batch / batch_size

    return {
        "ms_per_batch":   round(ms_per_batch,  3),
        "ms_per_sample":  round(ms_per_sample, 4),
        "ms_std":         round(float(np.std(times)), 3),
    }


def measure_all_archs(in_channels: int = 10,
                      device: torch.device = None,
                      batch_size: int = 32,
                      img_size: int = 64,
                      warmup_runs: int = 10,
                      measure_runs: int = 100,
                      pretrained: bool = False) -> pd.DataFrame:
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    assert device.type == "cuda", \
        "[measure_inference] GPU not available. Inference timing requires CUDA."

    records = []

    for arch_name in ARCH_REGISTRY.keys():
        print(f"  Measuring [{arch_name}]...", end=" ", flush=True)

        model  = build_arch(arch_name, in_channels=in_channels,
                            pretrained=pretrained)
        params = count_parameters(model)
        timing = measure_one_arch(model, in_channels, device,
                                  batch_size, img_size,
                                  warmup_runs, measure_runs)

        del model
        torch.cuda.empty_cache()

        records.append({
            "arch":           arch_name,
            "params_M":       round(params / 1e6, 2),
            "ms_per_batch":   timing["ms_per_batch"],
            "ms_per_sample":  timing["ms_per_sample"],
            "ms_std":         timing["ms_std"],
        })

        ms_per_batch_val = timing["ms_per_batch"]
        ms_per_sample_val = timing["ms_per_sample"]
        ms_std_val = timing["ms_std"]

        print(f"done. "
              f"{ms_per_batch_val:.2f} ms/batch "
              f"({ms_per_sample_val:.3f} ms/sample) "
              f"+/- {ms_std_val:.3f} ms  "
              f"| params={params/1e6:.1f}M")

    return pd.DataFrame(records)


if __name__ == "__main__":
    DEVICE     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    IN_CHANNELS = 10
    BATCH_SIZE  = 32

    print("=" * 60)
    print("Inference Time Measurement")
    print(f"Device     : {DEVICE}")
    print(f"Input      : ({BATCH_SIZE}, {IN_CHANNELS}, 64, 64)")
    print(f"Warmup     : 10 runs  |  Measure: 100 runs")
    print(f"Timer      : torch.cuda.Event (microsecond precision)")
    print("=" * 60 + "\n")

    df = measure_all_archs(
        in_channels  = IN_CHANNELS,
        device       = DEVICE,
        batch_size   = BATCH_SIZE,
        warmup_runs  = 10,
        measure_runs = 100,
        pretrained   = False,
    )

    print("\n[ Results ]\n")
    print(df.to_string(index=False))

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "inference_time.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_path}")