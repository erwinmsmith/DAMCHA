"""Measure CUDA peak allocated memory for decoder inference and backward passes."""
import argparse
import csv
from pathlib import Path
import torch
from scripts.profile_flops_latency import build
from models.baselines import BASELINE_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--tokens', type=int, default=64)
    parser.add_argument('--output', type=Path, default=Path('logs/experiments/memory.csv'))
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith('cuda'):
        parser.error('CUDA is required for CUDA memory measurements')
    if min(args.batch_size, args.tokens) < 1:
        parser.error('batch-size and tokens must be positive')
    torch.cuda.set_device(args.device)
    rows = []
    for tag in ['m_attention', 'standard', *BASELINE_NAMES]:
        model = build(tag, args.device)
        x = torch.randn(args.batch_size, args.tokens, 128, device=args.device)
        row = dict(model=tag, device=args.device, batch_size=args.batch_size, tokens=args.tokens)
        for training in [False, True]:
            model.train(training)
            model.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(args.device)
            with torch.set_grad_enabled(training):
                y = model(x)
                if training:
                    y.square().mean().backward()
            torch.cuda.synchronize(args.device)
            row['training_peak_mib' if training else 'inference_peak_mib'] = torch.cuda.max_memory_allocated(args.device) / 2**20
            del y
        rows.append(row); print(row)
        del model, x
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


if __name__ == '__main__':
    main()
