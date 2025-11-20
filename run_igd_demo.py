#!/usr/bin/env python3
import argparse
from pathlib import Path

from models import tfim, cft_1p1, syk_toy

def main():
    parser = argparse.ArgumentParser(description="IGD multimodel demo (TFIM, CFT, SYK toy)")
    parser.add_argument("--model", choices=["tfim", "cft", "syk"], default="tfim",
                        help="Which model to run: tfim, cft, syk")
    parser.add_argument("--mode", choices=["static", "dynamic", "both"], default="both",
                        help="Which part of the demo to run (syk only supports dynamic)")
    parser.add_argument("--figdir", default="figures", help="Output directory for figures")
    args = parser.parse_args()

    fig_dir = Path(args.figdir)

    if args.model == "tfim":
        tfim.run_demo(mode=args.mode, fig_dir=str(fig_dir))
    elif args.model == "cft":
        cft_1p1.run_demo(mode=args.mode, fig_dir=str(fig_dir))
    elif args.model == "syk":
        if args.mode == "static":
            print("[syk] static mode not implemented; running dynamic instead.")
        syk_toy.run_demo(fig_dir=str(fig_dir))
    else:
        raise ValueError(f"Unknown model: {{args.model}}")

if __name__ == "__main__":
    main()
