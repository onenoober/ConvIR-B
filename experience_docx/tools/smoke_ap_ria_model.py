#!/usr/bin/env python3
"""Minimal smoke test for AP-RIA model construction and identity initialization.

Run from repository root:
    python experience_docx/tools/smoke_ap_ria_model.py --device cpu
"""

import argparse
import os
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--version', default='base', choices=['small', 'base', 'large'])
    parser.add_argument('--data', default='Haze4K')
    parser.add_argument('--height', type=int, default=64)
    parser.add_argument('--width', type=int, default=64)
    args = parser.parse_args()

    repo_root = os.getcwd()
    sys.path.insert(0, os.path.join(repo_root, 'Dehazing', 'ITS'))

    from models.AP_RIAConvIR import build_net

    model = build_net(args.version, args.data, use_ap_ria=True).to(args.device)
    model.eval()

    x = torch.rand(1, 3, args.height, args.width, device=args.device)

    with torch.no_grad():
        # Adapter is zero-projection initialized, so adapted output should match
        # the anchor-side output at construction time up to numerical precision.
        outputs, aux = model(x, return_aux=True)
        max_abs = (outputs[-1] - aux['anchor_side']).abs().max().item()

    print('AP-RIA smoke test passed.')
    print('Final output shape:', tuple(outputs[-1].shape))
    print('Anchor identity max_abs:', max_abs)
    print('Gate low mean/detail mean:', aux['g_low'].mean().item(), aux['g_detail'].mean().item())

    # A loose threshold because different BasicConv implementations may include
    # normalization with small numerical effects; zero projection should make this exact.
    if max_abs > 1e-6:
        raise SystemExit('Identity initialization check failed: max_abs={}'.format(max_abs))


if __name__ == '__main__':
    main()
