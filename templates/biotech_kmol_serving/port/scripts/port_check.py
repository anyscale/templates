"""Parity check: ported (py3.11) model vs py3.9 kMoL reference logits."""
import argparse
import json

import numpy as np
import torch

import kmolport


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    ref = json.load(open(args.ref))
    smiles = ref["smiles"]
    ref_logits = np.array(ref["logits"])

    cfg = kmolport.load_config(args.config)
    model = kmolport.build_ensemble(cfg, checkpoint_dir=args.ckpt_dir, device=args.device)
    feat = kmolport.GraphFeaturizer()
    logits, var = kmolport.predict(model, feat, smiles, device=args.device)
    logits = logits.detach().cpu().numpy()

    diff = np.abs(logits - ref_logits)
    print(f"device={args.device}  torch={torch.__version__}  "
          f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}")
    print(f"logits shape={logits.shape}  ref shape={ref_logits.shape}")
    print(f"max abs diff={diff.max():.3e}  mean abs diff={diff.mean():.3e}")
    print("port logits[0][:5]=", logits[0][:5])
    print("ref  logits[0][:5]=", ref_logits[0][:5])
    ok = diff.max() < args.tol
    print(f"PARITY {'PASS' if ok else 'FAIL'} (tol={args.tol})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
