"""Generate the two printed labels (no brand names or logos). Run once; the PNGs are committed.

- stain_label.png: the gallon stain can's wrap-around label (528 x 166 mm), styled like a common\n  oil-based penetrating stain, without its brand name or logo.
- oily_waste_label.png: the oily-waste can's warning label (about 230 x 100 mm on the can), in the
  style such cans carry: yellow, black lettering, a flame pictogram, "keep lid closed".

Fonts: DejaVu Sans Bold and DejaVu Serif Bold (Bitstream Vera / DejaVu licence, free to use and redistribute), from the
system's /usr/share/fonts/truetype/dejavu.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
HERE = Path(__file__).resolve().parent


def _centred(d, W, text, y, size, fill, x0=0):
    f = ImageFont.truetype(FONT, size)
    w = d.textlength(text, font=f)
    d.text((x0 + (W - x0 - w) / 2, y), text, font=f, fill=fill)


def _teardrop(d, cx, cy, s, lean, fill):
    """Flame tongue: round base of radius 0.5 s centred at (cx, cy), tip 1.2 s above, leaning by `lean`."""
    import math
    r = 0.5 * s
    pts = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in [math.radians(t) for t in range(-20, 201, 10)]]
    pts.append((cx + lean * s, cy - 1.25 * s))
    d.polygon(pts, fill=fill)


def _flame(d, cx, cy, s, fill, bg):
    """Flammable-hazard flame: a large tongue with two smaller ones, a cut-out core and a bar beneath."""
    _teardrop(d, cx, cy + 0.25 * s, 0.75 * s, 0.10, fill)
    _teardrop(d, cx - 0.38 * s, cy + 0.45 * s, 0.45 * s, -0.25, fill)
    _teardrop(d, cx + 0.38 * s, cy + 0.45 * s, 0.45 * s, 0.25, fill)
    _teardrop(d, cx + 0.02 * s, cy + 0.50 * s, 0.32 * s, 0.05, bg)
    d.rectangle([cx - 0.85 * s, cy + 0.85 * s, cx + 0.85 * s, cy + 1.0 * s], fill=fill)


def _ghs_flame(d, cx, cy, r):
    """Red-bordered diamond with a black flame (the flammable-hazard pictogram)."""
    d.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=(200, 20, 20))
    k = 0.80 * r
    d.polygon([(cx, cy - k), (cx + k, cy), (cx, cy + k), (cx - k, cy)], fill=(255, 255, 255))
    _flame(d, cx, cy - 0.08 * r, 0.45 * r, (0, 0, 0), (255, 255, 255))


def _woodgrain(w, h, base, seed=3):
    """A stained-oak stripe: dark base with lighter and darker horizontal grain streaks."""
    import random
    rnd = random.Random(seed)
    img = Image.new("RGB", (w, h), base)
    d = ImageDraw.Draw(img)
    for _ in range(int(w * h / 900)):
        x, y = rnd.uniform(-40, w), rnd.uniform(0, h)
        length, thick = rnd.uniform(25, 160), rnd.choice((1, 1, 2, 3))
        k = rnd.uniform(-0.45, 0.35)
        c = tuple(max(0, min(255, int(v * (1 + k)))) for v in base)
        d.line([(x, y), (x + length, y + rnd.uniform(-2, 2))], fill=c, width=thick)
    return img


def stain_label(out=HERE / "stain_label.png"):
    """Wrap-around label in the trade dress of a common oil-based penetrating stain (navy top band,
    yellow body, serif title, a stripe of stained wood, red colour band), without any brand name or logo.
    The front panel is centred; the sides carry the fire warning. Wraps the whole can (360 degrees)."""
    W, H = 2400, 700
    navy, yellow, red, black = (22, 30, 58), (247, 205, 40), (196, 30, 36), (20, 20, 20)
    serif, sans = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", FONT
    img = Image.new("RGB", (W, H), yellow)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 150], fill=navy)
    d.rectangle([0, 150, W, 162], fill=(250, 250, 245))
    d.rectangle([0, H - 105, W, H], fill=red)
    d.rectangle([0, H - 117, W, H - 105], fill=(250, 250, 245))
    F0, F1 = 760, 1640          # the front panel's left and right edges (pixels)

    def text(t, cx, y, size, fill, font=sans):
        f = ImageFont.truetype(font, size)
        d.text((cx - d.textlength(t, font=f) / 2, y), t, font=f, fill=fill)

    cx = (F0 + F1) / 2
    text("PREMIUM OIL", cx, 22, 52, (240, 240, 240))
    text("STAIN & SEAL", cx, 82, 40, yellow)
    text("WOOD STAIN", cx, 195, 116, black, serif)
    text("Penetrating Oil-Based Stain", cx, 335, 50, black, serif)
    img.paste(_woodgrain(F1 - F0, 150, (78, 40, 18)), (int(F0), 405))
    text("SEMI-TRANSPARENT", F0 + 150, H - 140, 26, red)
    d.rounded_rectangle([cx - 250, H - 95, cx + 250, H - 15], radius=30, fill=(250, 250, 245))
    text("DARK WALNUT", cx, H - 82, 50, black)
    # Sides and back: the maker's spontaneous-combustion warning, as on real oil-stain cans.
    for x0 in (60, F1 + 120):
        y = 190
        for line, size in (("DANGER!", 70), ("RAGS, STEEL WOOL AND WASTE", 30), ("SOAKED WITH THIS PRODUCT", 30),
                           ("MAY SPONTANEOUSLY", 30), ("CATCH FIRE", 30), ("IF IMPROPERLY DISCARDED.", 30),
                           ("COMBUSTIBLE LIQUID", 30), ("1 GAL (3.78 L)", 40)):
            d.text((x0, y), line, font=ImageFont.truetype(sans, size), fill=black)
            y += size + 22
    img.save(out)
    return out


def oily_waste_label(out=HERE / "oily_waste_label.png"):
    W, H = 920, 400
    img = Image.new("RGB", (W, H), (250, 200, 10))
    d = ImageDraw.Draw(img)
    d.rectangle([8, 8, W - 9, H - 9], outline=(0, 0, 0), width=10)
    _ghs_flame(d, 150, 165, 115)
    _centred(d, W, "OILY", 30, 120, (0, 0, 0), x0=280)
    _centred(d, W, "WASTE", 150, 108, (0, 0, 0), x0=280)
    d.rectangle([24, 300, W - 25, H - 25], fill=(0, 0, 0))
    _centred(d, W, "KEEP LID CLOSED", 308, 58, (250, 200, 10))
    img.save(out)
    return out


if __name__ == "__main__":
    print(stain_label(), oily_waste_label())
