#!/usr/bin/env python3
"""AMD Compatible Badge — shields.io style, matches CI badge.

Usage:
  python3 scripts/amd_badge.py --output docs/amd-compatible.svg
  # Then embed in README:
  # ![AMD Compatible](docs/amd-compatible.svg)

The badge uses shields.io visual style for consistency with CI badges.
"""

import os


BADGE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="146" height="20" role="img" aria-label="AMD: COMPATIBLE&#10;verified on MI300X">
  <title>AMD Compatible ✓ — verified on MI300X</title>
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r">
    <rect width="146" height="20" rx="3" fill="#fff"/>
  </clipPath>
  <g clip-path="url(#r)">
    <rect width="52" height="20" fill="#ed1c24"/>
    <rect x="52" width="94" height="20" fill="#2ea44f"/>
    <rect width="146" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="'DejaVu Sans',Verdana,Geneva,sans-serif" font-size="11">
    <text x="26" y="14" font-weight="bold" letter-spacing=".5">AMD</text>
    <text x="98" y="14" font-weight="bold">COMPATIBLE ✓</text>
  </g>
</svg>"""


def main():
    out = "docs/amd-compatible.svg"
    os.makedirs("docs", exist_ok=True)
    with open(out, "w") as f:
        f.write(BADGE_SVG)
    print(f"✅ Badge: {out}")
    print(f"   Dimensions: 146x20 (matches shields.io)")


if __name__ == "__main__":
    main()
