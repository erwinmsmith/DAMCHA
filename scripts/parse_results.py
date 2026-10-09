"""Export the lowest-validation-loss epoch from each experiment CSV."""
import argparse
import csv
import math
from pathlib import Path


def collect_best(root):
    results = []
    for path in sorted(root.rglob('metrics_*.csv')):
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        valid = []
        for row in rows:
            try:
                loss = float(row.get('val_loss', ''))
            except (TypeError, ValueError):
                continue
            if math.isfinite(loss):
                valid.append((loss, row))
        if valid:
            _, best = min(valid, key=lambda pair: pair[0])
            results.append({'source': str(path), **best})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('outputs'))
    parser.add_argument('--output', type=Path, default=Path('logs/experiments/results.csv'))
    args = parser.parse_args()
    rows = collect_best(args.input)
    if not rows:
        parser.exit(1, 'No completed validation rows found.\n')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(f'Saved {len(rows)} experiments to {args.output}')


if __name__ == '__main__':
    main()
