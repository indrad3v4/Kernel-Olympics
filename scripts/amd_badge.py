#!/usr/bin/env python3
"""AMD Compatible Badge — generate an SVG badge for GitHub README.

Usage:
  python3 scripts/amd_badge.py --repo username/repo --output amd-compatible.svg

The badge shows "AMD Compatible ✅" with our MI300X proof.
Embed in README with:
  ![AMD Compatible](https://raw.githubusercontent.com/username/repo/main/docs/amd-compatible.svg)
"""

import argparse
import json
import hashlib
import os
import sys


BADGE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="220" height="28" viewBox="0 0 220 28">
  <defs>
    <linearGradient id="amd" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#ED1C24"/>
      <stop offset="100%" style="stop-color:#ED1C24"/>
    </linearGradient>
    <linearGradient id="ok" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#2EA44F"/>
      <stop offset="100%" style="stop-color:#2EA44F"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="100" height="28" fill="url(#amd)" rx="4" ry="4"/>
  <rect x="100" y="0" width="120" height="28" fill="url(#ok)" rx="4" ry="4"/>
  <text x="50" y="19" font-family="'DejaVu Sans',Arial,Helvetica,sans-serif"
        font-size="12" font-weight="bold" fill="#fff"
        text-anchor="middle" letter-spacing="0.5">AMD</text>
  <text x="160" y="19" font-family="'DejaVu Sans',Arial,Helvetica,sans-serif"
        font-size="12" font-weight="bold" fill="#fff"
        text-anchor="middle" letter-spacing="0.3">COMPATIBLE ✓</text>
</svg>"""


def generate_badge(repo: str, output: str) -> str:
    """Generate an AMD Compatible badge SVG."""
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        f.write(BADGE_SVG)
    return output


def main():
    parser = argparse.ArgumentParser(description="Generate AMD Compatible badge")
    parser.add_argument("--repo", help="GitHub repo name (e.g. user/repo)")
    parser.add_argument("--output", default="docs/amd-compatible.svg",
                        help="Output path for SVG badge")
    args = parser.parse_args()

    out = generate_badge(args.repo, args.output)
    print(f"✅ Badge generated: {out}")
    print(f"   Embed in README with:")
    print(f'   ![AMD Compatible]({out})')


if __name__ == "__main__":
    main()
