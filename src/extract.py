"""Split the source banner into animation layers (PNG with alpha)."""
from PIL import Image, ImageFilter
import numpy as np
from scipy import ndimage as ndi

SRC = 'assets/source.jpg'
OUT = 'assets/layers/'
import os; os.makedirs(OUT, exist_ok=True)

img = Image.open(SRC).convert('RGB')
a = np.asarray(img).astype(np.float32)
H, W, _ = a.shape
L = a @ np.array([0.299, 0.587, 0.114], np.float32)

def save_rgba(rgb, alpha, box, name):
    x0, y0, x1, y1 = box
    rgba = np.dstack([rgb[y0:y1, x0:x1], alpha[y0:y1, x0:x1] * 255]).clip(0, 255).astype(np.uint8)
    Image.fromarray(rgba, 'RGBA').save(OUT + name + '.png')
    print(name, box)

# ---------- cat ----------
cbox = (2250, 250, 3650, 1950)
x0, y0, x1, y1 = cbox
sub = a[y0:y1, x0:x1]
R, G, B = sub[..., 0], sub[..., 1], sub[..., 2]
bg = ((G >= 252) & (((G - B) > 8) | (R < G - 15))) | ((R > 250) & (G > 250) & (B > 250)) | (G > R + 20) | ((np.minimum(np.minimum(R, G), B) >= 243) & (G >= R - 1))
fg = ~bg
fg = ndi.binary_opening(fg, iterations=5)
lab, n = ndi.label(fg)
sizes = ndi.sum(fg, lab, range(1, n + 1))
fg = lab == (np.argmax(sizes) + 1)
fg = ndi.binary_closing(fg, iterations=6)
fg = ndi.binary_fill_holes(fg)
fg = ndi.binary_erosion(fg, iterations=1)
cat_alpha = np.zeros((H, W), np.float32)
cat_alpha[y0:y1, x0:x1] = ndi.gaussian_filter(fg.astype(np.float32), 1.6)
ys, xs = np.where(cat_alpha > 0.02)
cat_box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
save_rgba(a, cat_alpha, cat_box, 'cat')
CAT_BOX = list(map(int, cat_box))

# ---------- dark text / logos ----------
def text_alpha(box, hi, lo):
    m = np.zeros((H, W), np.float32)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = ((hi - L[y0:y1, x0:x1]) / (hi - lo)).clip(0, 1)
    return m

boxes = {
    'logo_startale': ((180, 280, 1280, 480), 215, 40),
    'logo_divider':  ((1300, 300, 1360, 460), 235, 150),
    'logo_pieverse': ((1390, 280, 2120, 480), 215, 40),
    'headline1':     ((180, 870, 1360, 1130), 215, 40),
    'headline2':     ((180, 1130, 2200, 1360), 215, 40),
    'subtitle':      ((180, 1500, 2400, 1620), 215, 40),
    'disclaimer':    ((160, 1905, 3700, 2060), 240, 130),
}
text_union = np.zeros((H, W), np.float32)
manifest = {}
for name, (box, hi, lo) in boxes.items():
    al = text_alpha(box, hi, lo)
    text_union = np.maximum(text_union, al)
    tb = np.where(al > 0.03)
    tight = (tb[1].min() - 4, tb[0].min() - 4, tb[1].max() + 5, tb[0].max() + 5)
    # keep the logo's original colour; text is pure black-ish
    col = np.zeros_like(a) + (np.array([20, 20, 20]) if name != 'disclaimer' else np.array([120, 120, 120]))
    save_rgba(col, al, tight, name)
    manifest[name] = list(map(int, tight))
    if name in ('headline1', 'headline2', 'subtitle'):
        x0, y0, x1, y1 = tight
        colsum = (al[y0:y1, x0:x1] > 0.05).any(0)
        gap_min = int(0.13 * (y1 - y0))
        words, start, run = [], None, 0
        for i, on in enumerate(list(colsum) + [False] * (gap_min + 1)):
            if on:
                if start is None: start = i
                run = 0; last = i
            elif start is not None:
                run += 1
                if run > gap_min:
                    words.append((start, last + 1)); start = None
        for k, (s0, s1) in enumerate(words):
            wb = (x0 + s0 - 3, y0, x0 + s1 + 3, y1)
            save_rgba(col, al, wb, f'{name}_w{k}')
            manifest.setdefault('words', {}).setdefault(name, []).append(list(map(int, wb)))

# ---------- clean background plate (inpaint text + cat) ----------
hole = (text_union > 0.02) | (cat_alpha > 0.01)
hole = ndi.binary_dilation(hole, iterations=10)
known = (~hole).astype(np.float32)
fill = a.copy()
# progressive normalized convolution: small → large sigma
small = Image.fromarray(a.astype(np.uint8)).resize((W // 4, H // 4), Image.BILINEAR)
sa = np.asarray(small).astype(np.float32)
sk = np.asarray(Image.fromarray((known * 255).astype(np.uint8)).resize((W // 4, H // 4), Image.BILINEAR)).astype(np.float32) / 255
sk = (sk > 0.99).astype(np.float32)
res = sa * sk[..., None]
wsum = sk.copy()
filled = sa.copy()
done = sk > 0
for s in [4, 8, 16, 32, 64, 128]:
    num = np.dstack([ndi.gaussian_filter(sa[..., c] * sk, s) for c in range(3)])
    den = ndi.gaussian_filter(sk, s)[..., None]
    est = num / np.maximum(den, 1e-6)
    newly = (~done) & (den[..., 0] > 0.15)
    filled[newly] = est[newly]
    done |= newly
filled[~done] = 255
big = np.asarray(Image.fromarray(filled.clip(0, 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)).astype(np.float32)
soft = ndi.gaussian_filter(hole.astype(np.float32), 6)[..., None]
plate = a * (1 - soft) + big * soft
Image.fromarray(plate.clip(0, 255).astype(np.uint8)).save(OUT + 'plate.jpg', quality=93)
print('plate done')

import json
manifest['cat'] = CAT_BOX
json.dump(manifest, open(OUT + 'manifest.json', 'w'), indent=1)
print({k: len(v) for k, v in manifest['words'].items()})
