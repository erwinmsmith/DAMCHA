"""
GPU memory profiler for all 6 models × 4 datasets.

Measures (all in MiB):
  params_MB      — model parameter memory
  peak_fwd_MB    — peak GPU memory, forward pass only   (inference)
  peak_train_MB  — peak GPU memory, forward + backward  (training)

Uses the exact batch shapes / model config from run_all_experiments.sh.
"""

import gc, csv, torch, torch.nn as nn
from pathlib import Path

from train       import build_model
from train_nlp   import build_nlp_model
from models.baselines import BASELINE_NAMES

# ── Common model config (matches main.py defaults + run_all_experiments.sh) ────
BASE_CFG = dict(
    d_model=128, n_heads=8, d_ff=256, n_layers=4,
    use_M=False, use_mlp=False, share_mlp=False,
    mlp_hidden=(), off_diag_mode='mlp',
    diag_scale=1.0, off_diag_scale=0.5, off_diag_alpha_init=0.1,
    use_layer_bias=True, layer_bias_rank=16,
    dev_mode=False, device='cuda',
)

M_ATTENTION_OVERRIDES = dict(use_M=True, use_mlp=True, share_mlp=True)

DATASETS = {
    # name  : (batch, seq_len, builder)
    'cifar10':   (64,  64,  'cv'),
    'cifar100':  (64,  64,  'cv'),
    'commongen': (32, 128, 'nlp'),
    'wmt':       (32, 256, 'nlp'),
}

MODELS = {
    'm_attention': dict(**M_ATTENTION_OVERRIDES),
    'tha':         dict(baseline='tha'),
    'dcmha':       dict(baseline='dcmha'),
    'colmha':      dict(baseline='colmha'),
    'mma':         dict(baseline='mma'),
    'moa':         dict(baseline='moa'),
}

DEVICE = 'cuda'


def build(model_overrides: dict, builder: str) -> nn.Module:
    cfg = {**BASE_CFG, **model_overrides}
    if builder == 'cv':
        return build_model(**cfg)
    else:
        return build_nlp_model(
            d_model=cfg['d_model'], n_heads=cfg['n_heads'],
            d_ff=cfg['d_ff'],      n_layers=cfg['n_layers'],
            use_M=cfg['use_M'],    use_mlp=cfg['use_mlp'],
            share_mlp=cfg['share_mlp'], mlp_hidden=cfg['mlp_hidden'],
            dev_mode=cfg['dev_mode'], device=cfg['device'],
            off_diag_mode=cfg['off_diag_mode'],
            diag_scale=cfg['diag_scale'],
            off_diag_scale=cfg['off_diag_scale'],
            off_diag_alpha_init=cfg['off_diag_alpha_init'],
            use_layer_bias=cfg['use_layer_bias'],
            layer_bias_rank=cfg['layer_bias_rank'],
            baseline=cfg.get('baseline'),
        )


def params_mb(model: nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2


def reset_mem():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def measure(model_tag: str, model_overrides: dict, ds_name: str,
            batch: int, seq_len: int, builder: str) -> dict:
    print(f"  {ds_name:12s} × {model_tag:14s} ...", end=" ", flush=True)

    # --- parameter memory ------------------------------------------------
    reset_mem()
    model = build(model_overrides, builder)
    pm = params_mb(model)

    # --- forward-only memory ---------------------------------------------
    reset_mem()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated() / 1024**2
    x = torch.randn(batch, seq_len, BASE_CFG['d_model'], device=DEVICE)
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
    torch.cuda.synchronize()
    peak_fwd = (torch.cuda.max_memory_allocated() / 1024**2) - base

    # --- train memory (fwd + bwd) ----------------------------------------
    reset_mem()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated() / 1024**2
    opt  = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x    = torch.randn(batch, seq_len, BASE_CFG['d_model'], device=DEVICE)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        out  = model(x)
        loss = out.mean()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    peak_train = (torch.cuda.max_memory_allocated() / 1024**2) - base

    del model, x, opt
    reset_mem()

    print(f"params={pm:.1f}MB  fwd={peak_fwd:.1f}MB  train={peak_train:.1f}MB")
    return dict(
        dataset=ds_name, model=model_tag,
        params_MB=round(pm, 2),
        peak_fwd_MB=round(peak_fwd, 2),
        peak_train_MB=round(peak_train, 2),
    )


# ── Run all 24 combinations ────────────────────────────────────────────────────
rows = []
for ds_name, (batch, seq_len, builder) in DATASETS.items():
    print(f"\n{'='*50}")
    print(f"Dataset: {ds_name}  (batch={batch}, seq={seq_len})")
    print(f"{'='*50}")
    for model_tag, overrides in MODELS.items():
        try:
            row = measure(model_tag, overrides, ds_name, batch, seq_len, builder)
            rows.append(row)
        except Exception as e:
            print(f"  ERROR: {e}")
            rows.append(dict(dataset=ds_name, model=model_tag,
                             params_MB='ERR', peak_fwd_MB='ERR', peak_train_MB='ERR'))

# ── Write CSV (one combined + one per dataset) ─────────────────────────────────
OUT = Path("logs/experiments")
HDR = ["dataset", "model", "params_MB", "peak_fwd_MB", "peak_train_MB"]

# Combined
with open(OUT / "results_memory.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=HDR)
    w.writeheader(); w.writerows(rows)

# Per-dataset
for ds in DATASETS:
    ds_rows = [r for r in rows if r["dataset"] == ds]
    with open(OUT / f"results_memory_{ds}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HDR)
        w.writeheader(); w.writerows(ds_rows)

print(f"\n\nResults written to {OUT}/results_memory*.csv")

# ── Pretty-print summary ───────────────────────────────────────────────────────
print(f"\n{'dataset':12s}  {'model':14s}  {'params(MB)':>10s}  {'fwd(MB)':>9s}  {'train(MB)':>10s}")
print("-" * 62)
for r in rows:
    print(f"{r['dataset']:12s}  {r['model']:14s}  {r['params_MB']:>10}  "
          f"{r['peak_fwd_MB']:>9}  {r['peak_train_MB']:>10}")
