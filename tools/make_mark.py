"""
Draw site/assets/mark.png, the browser tab icon.

Run once. The output is committed, so Pillow is a developer tool here and must
never appear in requirements.txt: Streamlit Community Cloud installs that file
into a 1 GB instance and the app does not draw anything at runtime.

The mark is the centre circle, the same pitch furniture the masthead uses. It
is deliberately not a logo. A one-person side project inventing a brand mark is
the kind of thing that makes a tool look like it is pretending to be a company.

    python tools/make_mark.py
"""
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "site" / "assets" / "mark.png"

GREEN = (31, 122, 61, 255)     # --pfl-green
GOLD = (216, 162, 34, 255)     # --pfl-gold
INK = (20, 20, 15, 255)        # --pfl-ink

# Drawn at 8x and resampled down: Pillow has no antialiased circle, so
# supersampling is the only way to get an edge that is not a staircase.
S = 8
SIZE = 64


def main():
    img = Image.new("RGBA", (SIZE * S, SIZE * S), INK)
    d = ImageDraw.Draw(img)

    pad = 9 * S
    d.ellipse([pad, pad, SIZE * S - pad, SIZE * S - pad],
              outline=GREEN, width=5 * S)

    spot = 22 * S
    d.ellipse([spot, spot, SIZE * S - spot, SIZE * S - spot], fill=GOLD)

    img = img.resize((SIZE, SIZE), Image.LANCZOS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, "PNG", optimize=True)
    print(f"wrote {OUT.relative_to(OUT.parent.parent.parent)} "
          f"({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
