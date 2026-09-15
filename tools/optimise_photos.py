"""
Shrink photographs for site/assets/photos/ until they are small enough to inline.

WHY THIS IS STRICT
------------------
Streamlit Community Cloud serves no static files, so app.py embeds these images
as base64 data URIs. That means every byte here is downloaded again on every
single page load, uncached, by a phone that may be on a metered bundle. A 2 MB
holiday photo would cost more than the entire rest of the page combined.

So this refuses to emit anything over MAX_KB. If a picture will not compress
that far at the target width it is dropped, loudly, rather than quietly shipped.

USAGE
-----
    python tools/optimise_photos.py ~/Pictures/some-folder
    python tools/optimise_photos.py path/to/one.jpg path/to/another.jpg

Each source becomes site/assets/photos/NN_some-caption.jpg, where the caption
comes from the original filename. app.py turns that filename into the visible
caption, so rename the output rather than editing any code.

Pillow is a developer dependency. It must not be added to requirements.txt.
"""
import sys
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "site" / "assets" / "photos"

MAX_KB = 120          # per image, after encoding
MAX_IMAGES = 4        # the strip shows four. More would just be weight.
QUALITIES = (82, 74, 66, 58, 50, 42)

# A BOX, not a width. The strip renders each photo at roughly 250x170 CSS
# pixels with object-fit: cover, so this is about 3x for a high density screen
# and everything beyond it is waste. Constraining width alone was the first
# version's mistake: a portrait photo scaled to 900 wide comes out 1200 tall,
# twice the pixels of a landscape one at the same setting, and it blew the
# size budget for a picture that gets cropped to a letterbox anyway.
TARGET_BOX = (800, 560)


def candidates(args):
    out = []
    for a in args:
        p = Path(a).expanduser()
        if p.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.png", "*.PNG"):
                out.extend(sorted(p.glob(ext)))
        elif p.exists():
            out.append(p)
        else:
            print(f"  skipped, not found: {p}")
    return out


def encode(img, path, quality):
    img.save(path, "JPEG", quality=quality, optimize=True, progressive=True)
    return path.stat().st_size


def main(args):
    if not args:
        print(__doc__)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    srcs = candidates(args)
    if not srcs:
        print("No images found.")
        return 1

    print(f"{len(srcs)} source image(s). Writing at most {MAX_IMAGES} into "
          f"{OUT.relative_to(ROOT)}, each under {MAX_KB} KB.\n")

    written = 0
    for src in srcs:
        if written >= MAX_IMAGES:
            print(f"  stopped at {MAX_IMAGES} images, the strip shows no more")
            break

        img = Image.open(src)
        # Phone photos carry an orientation flag that most viewers honour and
        # most naive resizes do not, which is how a portrait shot ends up on
        # its side on the web.
        img = ImageOps.exif_transpose(img).convert("RGB")

        img.thumbnail(TARGET_BOX, Image.LANCZOS)

        slug = "-".join(src.stem.lower().split())[:48]
        dest = OUT / f"{written + 1:02d}_{slug}.jpg"

        size = None
        for q in QUALITIES:
            size = encode(img, dest, q)
            if size <= MAX_KB * 1024:
                break

        if size > MAX_KB * 1024:
            dest.unlink(missing_ok=True)
            print(f"  DROPPED {src.name}: will not compress under {MAX_KB} KB "
                  f"(best was {size / 1024:.0f} KB). Crop it and try again.")
            continue

        written += 1
        print(f"  {dest.name:<44} {size / 1024:6.1f} KB  q={q}  "
              f"{img.width}x{img.height}")

    total = sum(p.stat().st_size for p in OUT.glob("*.jpg"))
    print(f"\n{written} image(s), {total / 1024:.0f} KB total inlined per page load.")
    if total > MAX_KB * 1024 * MAX_IMAGES:
        print("That is heavier than intended. Remove one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
