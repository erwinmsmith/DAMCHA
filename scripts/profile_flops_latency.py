"""Profile decoder forward passes with the repository's default configuration.

FLOPs cover operators supported by torch.profiler (matrix products/convolutions).
Latency is measured for the complete forward pass. Run from the repository root:
python -m scripts.profile_flops_latency --device cpu --repeats 5
"""
import argparse
import csv
import time
from pathlib import Path

import torch
from models import TransformerPC
from models.baselines import BaselineTransformer, BASELINE_NAMES


def build(model_tag, device):
    config = dict(d_model=128, n_heads=8, d_ff=256, n_layers=4, dropout=0.1)
    if model_tag in BASELINE_NAMES:
        model = BaselineTransformer(model_tag, **config)
    else:
        model = TransformerPC(**config, use_M=model_tag == 'm_attention',
                              use_mlp=model_tag == 'm_attention',
                              share_mlp=model_tag == 'm_attention', mlp_hidden=(256, 512))
    return model.to(device)


def profile_forward(model, batch, tokens, device, warmup, repeats):
    x = torch.randn(batch, tokens, model.d_model, device=device)
    model.eval()
    def synchronize():
        if str(device).startswith('cuda'):
            torch.cuda.synchronize(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        synchronize()
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(x)
            synchronize()
            times.append((time.perf_counter() - start) * 1000)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
            model(x)
    flops = sum(event.flops for event in prof.key_averages())
    latency = sum(times) / len(times)
    return dict(params=sum(p.numel() for p in model.parameters()),
                measured_gflops_per_sample=flops / batch / 1e9,
                latency_ms=latency, throughput_samples_s=batch * 1000 / latency)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--tokens', type=int, default=64)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--output', type=Path, default=Path('logs/experiments/profile.csv'))
    args = parser.parse_args()
    if min(args.batch_size, args.tokens, args.repeats) < 1 or args.warmup < 0:
        parser.error('batch-size, tokens and repeats must be positive; warmup must be nonnegative')
    rows = []
    for tag in ['m_attention', 'standard', *BASELINE_NAMES]:
        model = build(tag, args.device)
        row = dict(model=tag, device=args.device, batch_size=args.batch_size, tokens=args.tokens,
                   **profile_forward(model, args.batch_size, args.tokens, args.device, args.warmup, args.repeats))
        rows.append(row)
        print(row)
        del model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


if __name__ == '__main__':
    main()
