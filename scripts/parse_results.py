"""
Parse all experiment logs in logs/experiments/ and write one CSV per dataset
containing the best-epoch metrics for each model.
"""
import re, csv, sys, traceback
from pathlib import Path

LOG_DIR = Path("logs/experiments")
MODEL_ORDER = ["m_attention", "tha", "dcmha", "colmha", "mma", "moa"]

print(f"LOG_DIR   : {LOG_DIR.resolve()}")
print(f"DIR exists: {LOG_DIR.exists()}")
print(f"Log files : {[p.name for p in sorted(LOG_DIR.glob('*.log'))]}\n")


def flat(path: Path) -> str:
    """Read file and strip all newlines so wrapped long lines are rejoined."""
    return path.read_text(errors="replace").replace("\n", "")


# ── CV regex patterns ──────────────────────────────────────────────────────────
RCV_VAL = re.compile(
    r"Validation - Loss:\s*([\d.]+),\s*MSE:\s*([\d.]+)\s*"
    r"\[Q25=([\d.]+),\s*Q75=([\d.]+)\],\s*FID:\s*([-\d.]+)\s*"
    r"\[Q25=([-\d.]+),\s*Q75=([-\d.]+)\]"
)
RCV_TOP50 = re.compile(
    r"Top-50 - MSE samples FID:\s*([-\d.]+),\s*FID samples MSE:\s*([-\d.]+)"
)
RCV_BEST = re.compile(
    r"Best Single - MSE:\s*([-\d.]+),\s*FID:\s*([-\d.]+)"
)

# ── NLP regex patterns ─────────────────────────────────────────────────────────
RNLP_VAL = re.compile(r"Validation - Loss:\s*([\d.]+)")
RNLP_NLL = re.compile(
    r"NLL\s*:\s*mean=([-\d.]+)\s+Q25=([-\d.]+)\s+Q75=([-\d.]+)\s+top50=([-\d.]+)"
)
RNLP_BART = re.compile(
    r"BARTScore:\s*mean=([-\d.]+)\s+Q25=([-\d.]+)\s+Q75=([-\d.]+)\s+top50=([-\d.]+)"
)


def parse_cv(path: Path):
    text   = flat(path)
    vals   = list(RCV_VAL.finditer(text))
    top50s = list(RCV_TOP50.finditer(text))
    bests  = list(RCV_BEST.finditer(text))
    print(f"    {path.name}: found {len(vals)} val, {len(top50s)} top50, {len(bests)} best")
    epochs = []
    for vm in vals:
        vend = vm.end()
        t = next((m for m in top50s if m.start() > vend), None)
        b = next((m for m in bests  if m.start() > vend), None)
        if t is None or b is None:
            continue
        epochs.append(dict(
            val_loss      = float(vm[1]),
            mse_mean=float(vm[2]), mse_q25=float(vm[3]), mse_q75=float(vm[4]),
            fid_mean=float(vm[5]), fid_q25=float(vm[6]), fid_q75=float(vm[7]),
            top50_mse_fid = float(t[1]),
            top50_fid_mse = float(t[2]),
            best_mse      = float(b[1]),
            best_fid      = float(b[2]),
        ))
    return min(epochs, key=lambda e: e["val_loss"]) if epochs else None


def parse_nlp(path: Path):
    text  = flat(path)
    vals  = list(RNLP_VAL.finditer(text))
    nlls  = list(RNLP_NLL.finditer(text))
    barts = list(RNLP_BART.finditer(text))
    print(f"    {path.name}: found {len(vals)} val, {len(nlls)} nll, {len(barts)} bart")
    epochs = []
    for vm in vals:
        vp = vm.start()
        n  = next((m for m in nlls  if m.start() > vp), None)
        if n is None:
            continue
        b  = next((m for m in barts if m.start() > n.start()), None)
        if b is None:
            continue
        epochs.append(dict(
            val_loss   = float(vm[1]),
            nll_mean=float(n[1]), nll_q25=float(n[2]),
            nll_q75=float(n[3]),  nll_top50=float(n[4]),
            bart_mean=float(b[1]), bart_q25=float(b[2]),
            bart_q75=float(b[3]),  bart_top50=float(b[4]),
        ))
    return min(epochs, key=lambda e: e["val_loss"]) if epochs else None


# ── CV tables ──────────────────────────────────────────────────────────────────
CV_HDR = [
    "model", "best_val_loss",
    "mse_mean", "mse_q25", "mse_q75",
    "fid_mean", "fid_q25", "fid_q75",
    "top50_mse->fid", "top50_fid->mse",
    "best_single_mse", "best_single_fid",
]

for ds in ["cifar10", "cifar100"]:
    print(f"\n=== {ds} ===")
    rows = []
    for mdl in MODEL_ORDER:
        f = LOG_DIR / f"{ds}_{mdl}.log"
        if not f.exists():
            print(f"  MISSING: {f.name}"); continue
        try:
            d = parse_cv(f)
        except Exception:
            print(f"  ERROR parsing {f.name}:"); traceback.print_exc(); continue
        if d is None:
            print(f"  NO MATCH: {f.name}"); continue
        rows.append([
            mdl,
            f"{d['val_loss']:.4f}",
            f"{d['mse_mean']:.4f}", f"{d['mse_q25']:.4f}", f"{d['mse_q75']:.4f}",
            f"{d['fid_mean']:.4f}", f"{d['fid_q25']:.4f}", f"{d['fid_q75']:.4f}",
            f"{d['top50_mse_fid']:.4f}", f"{d['top50_fid_mse']:.6f}",
            f"{d['best_mse']:.6f}",      f"{d['best_fid']:.4f}",
        ])
    out = LOG_DIR / f"results_{ds}.csv"
    with open(out, "w", newline="") as fh:
        csv.writer(fh).writerows([CV_HDR] + rows)
    print(f"  → Wrote {out}  ({len(rows)} rows)")


# ── NLP tables ─────────────────────────────────────────────────────────────────
NLP_HDR = [
    "model", "best_val_loss",
    "nll_mean", "nll_q25", "nll_q75", "nll_top50",
    "bartscore_mean", "bartscore_q25", "bartscore_q75", "bartscore_top50",
]

for ds in ["commongen", "wmt"]:
    print(f"\n=== {ds} ===")
    rows = []
    for mdl in MODEL_ORDER:
        f = LOG_DIR / f"{ds}_{mdl}.log"
        if not f.exists():
            print(f"  MISSING: {f.name}"); continue
        try:
            d = parse_nlp(f)
        except Exception:
            print(f"  ERROR parsing {f.name}:"); traceback.print_exc(); continue
        if d is None:
            print(f"  NO MATCH: {f.name}"); continue
        rows.append([
            mdl,
            f"{d['val_loss']:.4f}",
            f"{d['nll_mean']:.4f}", f"{d['nll_q25']:.4f}",
            f"{d['nll_q75']:.4f}",  f"{d['nll_top50']:.4f}",
            f"{d['bart_mean']:.4f}", f"{d['bart_q25']:.4f}",
            f"{d['bart_q75']:.4f}",  f"{d['bart_top50']:.4f}",
        ])
    out = LOG_DIR / f"results_{ds}.csv"
    with open(out, "w", newline="") as fh:
        csv.writer(fh).writerows([NLP_HDR] + rows)
    print(f"  → Wrote {out}  ({len(rows)} rows)")

print("\nDone.")
