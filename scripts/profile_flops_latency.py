"""
Analytical FLOPs + GPU latency profiler — v2 (correct).

Why thop (v1) was wrong:
  • Missed ALL torch.matmul/einsum ops (QK^T, attn@V, xM_heads, MoA einsum)
  • MLP-M uses Linear(D, dim_M) on input [1, D] (seq-independent),
    but thop's hook was left live and fired on subsequent forward passes
    causing AttributeError with PyTorch 2.7.

This script uses closed-form MACs for every operation in each model,
then multiplies by 2 to get FLOPs.

Config: D=128, H=8, d_h=16, F=256, n_layers=4  (main.py defaults)
"""

import gc, csv, time, math, torch
from pathlib import Path
from train     import build_model
from train_nlp import build_nlp_model

# ── Config ────────────────────────────────────────────────────────────────────
D, H, F_FF, L = 128, 8, 256, 4   # d_model, n_heads, d_ff, n_layers
d_h = D // H                       # 16
DATASETS = {
    'cifar10':   (64,  64,  'cv'),
    'cifar100':  (64,  64,  'cv'),
    'commongen': (32, 128, 'nlp'),
    'wmt':       (32, 256, 'nlp'),
}
DEVICE  = 'cuda'
WARMUP  = 30
REPEATS = 200

BASE_CFG = dict(
    d_model=D, n_heads=H, d_ff=F_FF, n_layers=L,
    use_M=False, use_mlp=False, share_mlp=False, mlp_hidden=(),
    off_diag_mode='mlp', diag_scale=1.0, off_diag_scale=0.5,
    off_diag_alpha_init=0.1, use_layer_bias=True, layer_bias_rank=16,
    dev_mode=False, device=DEVICE,
)


# ── Analytical FLOPs (MACs × 2 → FLOPs) ─────────────────────────────────────

def _std_layer_macs(T):
    """Standard MHA + FFN MACs per decoder layer."""
    qkvo  = 4 * T * D * D          # Q, K, V, O  linear projections
    qk_t  = H * T * T * d_h        # QK^T  (= T²·D)
    av    = H * T * T * d_h        # attn @ V
    ffn   = 2 * T * D * F_FF       # FFN up + down
    return qkvo + qk_t + av + ffn

def analytical_macs(model_tag, T):
    """Return total MACs for one forward pass (batch=1)."""
    base = _std_layer_macs(T) * L

    if model_tag == 'tha':
        # Two H×H linear ops on [T², H]: pre- and post-softmax mixing
        extra = 2 * T * T * H * H * L
        return base + extra

    elif model_tag == 'dcmha':
        # router_rank = max(4, H//2) = 4
        rk = max(4, H // 2)
        router_macs = (T * D * rk + T * rk * (H * H)) * L          # 2-layer router
        gate_macs   = T * D * H * L                                 # gate_proj
        compose_macs = T * T * H * H * L                           # einsum composition
        return base + router_macs + gate_macs + compose_macs

    elif model_tag == 'colmha':
        # Shared Q/K bases (same cost as standard Q/K), plus
        # combine_q/combine_k einsums: H×H applied over T×d_h
        combine = 2 * T * H * H * d_h * L  # einsum('oi,btid->btod'): H×H×T×d_h each
        return base + combine

    elif model_tag == 'mma':
        # Same arithmetic as standard (masking is index ops, zero MACs)
        return base

    elif model_tag == 'moa':
        # n_experts = 2H; all experts compute attention; top-k gather is free
        n_e = 2 * H
        # Replace standard Q+K+V (3×T×D²) with 2H-expert Q+K+V
        std_qkv = 3 * T * D * D                       # standard
        moa_qkv = 3 * T * n_e * D * d_h               # 2H experts, each D→d_h
        # Replace standard QK^T+attn@V (2×T²×D) with all-expert versions
        std_attn = 2 * H * T * T * d_h
        moa_attn = 2 * n_e * T * T * d_h
        # O proj: top-k×d_h → D  (k = H)
        std_o    = T * D * D
        moa_o    = T * (H * d_h) * D                  # == T×D² (same)
        # Router: Linear(D, n_e)
        router   = T * D * n_e
        # per layer delta
        delta = (moa_qkv - std_qkv + moa_attn - std_attn + router) * L
        return base + delta

    elif model_tag in ('m_attention', 'm_attention_compact4'):
        # NO Q, K projections; instead:
        #   V, O projections  (2 × T×D²)
        #   xM_heads = x @ M_blocks  (H×T×d_h×d_h per layer)
        #   logits = xM @ x.T        (H×T×T×d_h)
        #   attn @ V                 (H×T×T×d_h)
        #   FFN                      (2T×D×F)
        per_layer = (
            2 * T * D * D +           # V, O projections
            H * T * d_h * d_h +       # xM_heads  (T×D²/H)
            H * T * T * d_h +         # logits
            H * T * T * d_h +         # attn @ V
            2 * T * D * F_FF          # FFN
        )
        seq_macs = per_layer * L

        # MLP-M: seq-independent, runs once per layer on [1, D] input
        if model_tag == 'm_attention':
            # Linear(D, D²=16384) + layer_bias [D,r]@[r,D]
            mlp_per_call  = D * (D * D) + D * 16 * D    # ~2.36M MACs
        else:
            # compact_rank=4: Linear(D, 2*D*r=1024) + layer_bias
            mlp_per_call  = D * (2 * D * 4) + D * 16 * D   # ~0.39M MACs
        mlp_total = mlp_per_call * L

        return seq_macs + mlp_total

    else:
        raise ValueError(f"Unknown model: {model_tag}")


# ── Model builder ─────────────────────────────────────────────────────────────

def build(model_tag, builder):
    cfg = dict(BASE_CFG)
    if model_tag.startswith('m_attention'):
        cfg.update(use_M=True, use_mlp=True, share_mlp=True)
        if 'compact4' in model_tag:
            cfg['compact_rank'] = 4
    elif model_tag != 'standard':
        cfg['baseline'] = model_tag

    if builder == 'cv':
        return build_model(**cfg)
    return build_nlp_model(
        d_model=cfg['d_model'], n_heads=cfg['n_heads'],
        d_ff=cfg['d_ff'],       n_layers=cfg['n_layers'],
        use_M=cfg['use_M'],     use_mlp=cfg['use_mlp'],
        share_mlp=cfg['share_mlp'], mlp_hidden=cfg['mlp_hidden'],
        dev_mode=cfg['dev_mode'], device=cfg['device'],
        off_diag_mode=cfg['off_diag_mode'],
        diag_scale=cfg['diag_scale'],
        off_diag_scale=cfg['off_diag_scale'],
        off_diag_alpha_init=cfg['off_diag_alpha_init'],
        use_layer_bias=cfg['use_layer_bias'],
        layer_bias_rank=cfg['layer_bias_rank'],
        baseline=cfg.get('baseline'),
        compact_rank=cfg.get('compact_rank', 0),
    )


# ── Latency (GPU, full batch) ─────────────────────────────────────────────────

def measure_latency(model_tag, batch, seq_len, builder):
    model = build(model_tag, builder)
    model.eval()
    xb = torch.randn(batch, seq_len, D, device=DEVICE)

    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(xb)
    torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(REPEATS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(xb)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    lat = sum(times) / len(times)
    std = (sum((t - lat) ** 2 for t in times) / len(times)) ** 0.5
    tput = batch / (lat / 1000)

    del model, xb
    gc.collect(); torch.cuda.empty_cache()
    return lat, std, tput


# ── Main loop ─────────────────────────────────────────────────────────────────

MODELS = ['m_attention', 'm_attention_compact4',
          'tha', 'dcmha', 'colmha', 'mma', 'moa']

rows = []
for ds_name, (batch, seq_len, builder) in DATASETS.items():
    print(f"\n{'='*62}")
    print(f"Dataset: {ds_name}  (batch={batch}, seq={seq_len})")
    print(f"{'='*62}")
    for mt in MODELS:
        print(f"  {ds_name:12s} × {mt:22s} ...", end=" ", flush=True)
        try:
            macs     = analytical_macs(mt, seq_len)
            gflops   = macs * 2 / 1e9              # MACs → FLOPs
            params_M = sum(p.numel() for p in build(mt, builder).parameters()) / 1e6
            gc.collect(); torch.cuda.empty_cache()
            lat, std_lat, tput = measure_latency(mt, batch, seq_len, builder)
            print(f"params={params_M:.2f}M  GFLOPs={gflops:.4f}  "
                  f"lat={lat:.2f}±{std_lat:.2f}ms  tput={tput:.0f}sps")
            rows.append(dict(
                dataset=ds_name, model=mt,
                params_M=round(params_M, 3),
                gflops=round(gflops, 4),
                lat_mean_ms=round(lat, 3),
                lat_std_ms=round(std_lat, 3),
                throughput_sps=round(tput, 1),
            ))
        except Exception as e:
            import traceback; traceback.print_exc()
            rows.append(dict(dataset=ds_name, model=mt,
                             params_M='ERR', gflops='ERR',
                             lat_mean_ms='ERR', lat_std_ms='ERR', throughput_sps='ERR'))

# ── Write CSV ─────────────────────────────────────────────────────────────────
OUT = Path("logs/experiments")
HDR = ["dataset", "model", "params_M", "gflops",
       "lat_mean_ms", "lat_std_ms", "throughput_sps"]

with open(OUT / "results_flops_latency_v2.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=HDR); w.writeheader(); w.writerows(rows)

for ds in DATASETS:
    ds_rows = [r for r in rows if r["dataset"] == ds]
    with open(OUT / f"results_flops_latency_v2_{ds}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HDR); w.writeheader(); w.writerows(ds_rows)

print(f"\nResults → {OUT}/results_flops_latency_v2*.csv\n")

# ── Pretty summary ────────────────────────────────────────────────────────────
print(f"\n{'dataset':12s}  {'model':24s}  {'params(M)':>9s}  {'GFLOPs':>8s}"
      f"  {'lat(ms)':>8s}  {'±std':>6s}  {'tput(sps)':>10s}")
print("-" * 85)
for r in rows:
    print(f"{r['dataset']:12s}  {r['model']:24s}  {r['params_M']:>9}  {r['gflops']:>8}"
          f"  {r['lat_mean_ms']:>8}  {r['lat_std_ms']:>6}  {r['throughput_sps']:>10}")
