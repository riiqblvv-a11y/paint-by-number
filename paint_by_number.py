"""
Paint by Number Generator — Core Logic
=======================================
Import this module from app.py (Gradio) or use CLI directly.

Usage (CLI):
    python paint_by_number.py myimage.jpg --colors 14 --scale 8
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin
from scipy.ndimage import label as scipy_label, binary_dilation
import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec


# ── helpers ───────────────────────────────────────────────────────────────────

def load_image(path):
    if path:
        p = Path(path)
        if not p.exists():
            sys.exit(f"File not found: {path}")
        return Image.open(p).convert("RGB"), p
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        chosen = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp *.gif *.tiff")]
        )
        if not chosen:
            sys.exit("No file selected.")
        p = Path(chosen)
        return Image.open(p).convert("RGB"), p
    except Exception:
        sys.exit("Provide an image path: python paint_by_number.py myimage.jpg")


def resize_for_processing(img, grid_size):
    w, h = img.size
    if w >= h:
        new_w, new_h = grid_size, max(1, round(grid_size * h / w))
    else:
        new_h, new_w = grid_size, max(1, round(grid_size * w / h))
    return img.resize((new_w, new_h), Image.LANCZOS)


def auto_grid_size(img, requested_size=None):
    if requested_size is not None:
        return requested_size
    img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img_gray, 50, 150)
    complexity = edges.sum() / (img_gray.shape[0] * img_gray.shape[1])
    if complexity > 0.15:   size = 200
    elif complexity > 0.08: size = 160
    else:                   size = 130
    print(f"  Auto grid size : {size}  (complexity={complexity:.3f})")
    return size


def load_font(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for fp in candidates:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            pass
    return ImageFont.load_default()


def rgb_to_hex(r, g, b):
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def color_name(r, g, b):
    r, g, b = r / 255, g / 255, b / 255
    mx, mn = max(r, g, b), min(r, g, b)
    l = (mx + mn) / 2
    sat = 0 if mx == mn else (
        (mx - mn) / (2 - mx - mn) if l > 0.5 else (mx - mn) / (mx + mn)
    )
    if sat < 0.12:
        if l < 0.15: return "Charcoal"
        if l < 0.35: return "Dark gray"
        if l < 0.60: return "Gray"
        if l < 0.82: return "Light gray"
        return "White"
    if mx == r:   hh = ((g - b) / (mx - mn)) % 6
    elif mx == g: hh = (b - r) / (mx - mn) + 2
    else:         hh = (r - g) / (mx - mn) + 4
    hh *= 60
    if hh < 15:    base = "red"
    elif hh < 40:  base = "orange"
    elif hh < 65:  base = "yellow"
    elif hh < 80:  base = "yellow-green"
    elif hh < 150: base = "green"
    elif hh < 185: base = "cyan"
    elif hh < 255: base = "blue"
    elif hh < 290: base = "violet"
    elif hh < 330: base = "magenta"
    elif hh < 350: base = "pink"
    else:          base = "red"
    if l < 0.20: return f"Very dark {base}"
    if l < 0.38: return f"Dark {base}"
    if l > 0.75: return f"Light {base}"
    if l > 0.60: return f"Pale {base}"
    return base.capitalize()


# ── processing pipeline ───────────────────────────────────────────────────────

def apply_clahe(img_cv):
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    img_cv = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_Lab2BGR)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def get_saliency_map(img_cv):
    h, w = img_cv.shape[:2]
    try:
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, sal_map = saliency.computeSaliency(img_cv)
        if success:
            sal_map = cv2.resize(sal_map.astype(np.float32), (w, h))
            sal_map = cv2.GaussianBlur(sal_map, (15, 15), 0)
            sal_map -= sal_map.min()
            sal_map /= (sal_map.max() + 1e-8)
            return sal_map
    except Exception:
        pass
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    sal_map = cv2.GaussianBlur(edges, (31, 31), 0)
    sal_map -= sal_map.min()
    sal_map /= (sal_map.max() + 1e-8)
    return sal_map


def get_diverse_seeds(arr_lab, n_colors):
    arr_uint8 = np.clip(arr_lab, 0, 255).astype(np.uint8).reshape(1, -1, 3)
    arr_bgr   = cv2.cvtColor(arr_uint8, cv2.COLOR_Lab2BGR)
    arr_hsv   = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hue = arr_hsv[:, 0].astype(np.float32)
    sat = arr_hsv[:, 1].astype(np.float32)
    seeds = []
    bin_edges = np.linspace(0, 180, n_colors + 1)
    for i in range(n_colors):
        mask = (hue >= bin_edges[i]) & (hue < bin_edges[i + 1])
        candidates = arr_lab[mask]
        if len(candidates) > 0:
            sat_w = sat[mask] / (sat[mask].sum() + 1e-8)
            idx = np.searchsorted(np.cumsum(sat_w), np.random.rand())
            seeds.append(candidates[min(idx, len(candidates) - 1)])
        else:
            seeds.append(arr_lab[np.random.randint(len(arr_lab))])
    return np.array(seeds, dtype=np.float32)


def get_rare_and_vivid_pixels(arr_lab, img_cv):
    coarse = (arr_lab / 16).astype(np.int32)
    keys = coarse[:, 0] * 10000 + coarse[:, 1] * 100 + coarse[:, 2]
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    rarity = 1.0 / (counts[inverse] + 1)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    saturation = hsv[:, 1].astype(np.float32) / 255.0
    score = rarity * saturation
    return score >= np.percentile(score, 85)


def remove_ghost_clusters(arr, labels, centers, min_pixel_fraction=0.002):
    min_pixels = max(1, int(len(labels) * min_pixel_fraction))
    counts = np.bincount(labels, minlength=len(centers))
    alive  = np.where(counts >= min_pixels)[0]
    dead   = np.where(counts < min_pixels)[0]
    if len(dead) == 0:
        return labels, centers
    print(f"  Removing {len(dead)} ghost clusters (< {min_pixels} px each)")
    alive_centers = centers[alive]
    new_labels    = pairwise_distances_argmin(arr, alive_centers)
    return new_labels, alive_centers


def quantize_image(img, n_colors):
    img_cv  = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_cv  = apply_clahe(img_cv)
    sal_map = get_saliency_map(img_cv)

    img_lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2Lab)
    h, w, _ = img_lab.shape
    arr     = img_lab.reshape(-1, 3).astype(np.float32)

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = cv2.GaussianBlur(
        (edges > 0).astype(np.float32), (15, 15), 0
    )

    sal_flat = cv2.resize(sal_map, (w, h)).flatten()

    rare_mask    = get_rare_and_vivid_pixels(arr, img_cv)
    salient_mask = sal_flat >= np.percentile(sal_flat, 70)
    edge_mask    = edges.flatten() > 0

    detail_train = np.vstack([
        arr,
        arr[rare_mask], arr[rare_mask], arr[rare_mask],
        arr[rare_mask], arr[rare_mask],
        arr[salient_mask], arr[salient_mask], arr[salient_mask],
        arr[edge_mask], arr[edge_mask],
    ]).astype(np.float32)

    seeds    = get_diverse_seeds(arr, n_colors)
    n_global = max(4, int(n_colors * 0.5))
    n_detail = n_colors - n_global

    km_global = MiniBatchKMeans(n_clusters=n_global, n_init=1,
                                 init=seeds[:n_global], random_state=42)
    km_global.fit(arr)

    km_detail = MiniBatchKMeans(n_clusters=n_detail, n_init=1,
                                 init=seeds[n_global:], random_state=99)
    km_detail.fit(detail_train)

    merged = np.vstack([km_global.cluster_centers_,
                        km_detail.cluster_centers_])
    labels = pairwise_distances_argmin(arr, merged)
    labels, merged = remove_ghost_clusters(arr, labels, merged,
                                            min_pixel_fraction=0.002)

    n_actual    = len(merged)
    centers_lab = merged.reshape(1, n_actual, 3).astype(np.uint8)
    palette     = cv2.cvtColor(centers_lab, cv2.COLOR_Lab2RGB).reshape(n_actual, 3)
    label_grid  = labels.reshape(h, w).astype(np.int32)
    return palette, label_grid, edge_density


def smooth_label_grid_adaptive(label_grid, edge_density, passes=1):
    h, w = label_grid.shape
    density      = cv2.resize(edge_density, (w, h))
    complex_mask = density > 0.3
    grid = label_grid.astype(np.uint8)
    for _ in range(passes):
        smoothed = cv2.medianBlur(grid, 3)
        grid = np.where(complex_mask, grid, smoothed).astype(np.uint8)
    return grid.astype(np.int32)


def flood_fill_small_regions(label_grid, min_size=30):
    """n_colors derived from label_grid so it stays in sync after ghost removal."""
    n_colors = int(label_grid.max()) + 1
    grid = label_grid.copy()
    for c in range(n_colors):
        mask = (grid == c)
        labeled, num = scipy_label(mask)
        for i in range(1, num + 1):
            region = labeled == i
            if region.sum() >= min_size:
                continue
            dilated       = binary_dilation(region)
            border_labels = grid[dilated & ~region]
            if len(border_labels) > 0:
                grid[region] = np.bincount(
                    border_labels, minlength=n_colors
                ).argmax()
    return grid


# ── rendering ─────────────────────────────────────────────────────────────────

def draw_smooth_contours(draw, label_grid, scale):
    n_colors = int(label_grid.max()) + 1
    h, w     = label_grid.shape
    for c in range(n_colors):
        mask     = ((label_grid == c).astype(np.uint8) * 255)
        big_mask = cv2.resize(mask, (w * scale, h * scale),
                               interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(
            big_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
        )
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, epsilon=0.1, closed=True)
            if len(approx) < 3:
                continue
            pts = [(int(p[0][0]), int(p[0][1])) for p in approx]
            draw.line(pts + [pts[0]], fill=(0, 0, 0), width=1)


def place_numbers(draw, label_grid, scale, font):
    n_colors   = int(label_grid.max()) + 1
    min_pixels = 40
    for c in range(n_colors):
        mask = (label_grid == c)
        labeled, num = scipy_label(mask)
        for i in range(1, num + 1):
            region = labeled == i
            if region.sum() < min_pixels:
                continue
            ys, xs = np.where(region)
            cy     = int(np.mean(ys)) * scale + scale // 2
            cx     = int(np.mean(xs)) * scale + scale // 2
            text   = str(c + 1)
            bbox   = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((cx - tw // 2, cy - th // 2),
                      text, fill=(20, 20, 20), font=font)


def build_pbn_image(label_grid, palette, scale=8):
    h, w       = label_grid.shape
    big_labels = np.repeat(np.repeat(label_grid, scale, axis=0), scale, axis=1)

    pale_factor  = 0.35
    pale_palette = (
        palette.astype(float) * pale_factor + 255 * (1 - pale_factor)
    ).astype(np.uint8)

    img  = Image.fromarray(pale_palette[big_labels], "RGB")
    draw = ImageDraw.Draw(img)
    draw_smooth_contours(draw, label_grid, scale)
    place_numbers(draw, label_grid, scale,
                  load_font(max(9, min(16, scale * 2))))
    return img


# ── palette legend image ──────────────────────────────────────────────────────

def build_palette_legend(palette):
    n    = len(palette)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 2.8, rows * 1.6 + 0.6))
    fig.patch.set_facecolor("#f8f8f8")
    fig.suptitle("Color Palette", fontsize=13, fontweight="bold", y=1.01)
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]
    for i, ax in enumerate(axes_flat):
        if i < n:
            r, g, b = palette[i]
            hex_c   = rgb_to_hex(r, g, b)
            name    = color_name(r, g, b)
            ax.set_facecolor(hex_c)
            s2 = pe.withStroke(linewidth=2,   foreground="black")
            s1 = pe.withStroke(linewidth=1.5, foreground="black")
            ax.text(0.5, 0.62, str(i + 1), ha="center", va="center",
                    fontsize=16, fontweight="bold", color="white",
                    transform=ax.transAxes, path_effects=[s2])
            ax.text(0.5, 0.28, name, ha="center", va="center",
                    fontsize=7.5, color="white",
                    transform=ax.transAxes, path_effects=[s1])
            ax.text(0.5, 0.06, hex_c, ha="center", va="bottom",
                    fontsize=6.5, color="white", fontfamily="monospace",
                    transform=ax.transAxes, path_effects=[s1])
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#ccc"); spine.set_linewidth(0.8)
        else:
            ax.set_visible(False)
    plt.tight_layout()
    return fig


# ── main entry point (returns PIL images, usable by CLI and Gradio) ───────────

def run(img: Image.Image, n_colors=14, scale=8, smooth=1, grid_size=None):
    """
    Main pipeline. Accepts a PIL image, returns (pbn_image, palette_fig).
    This is the single entry point for both CLI and Gradio.
    """
    grid_size = auto_grid_size(img, grid_size)
    grid_size = max(64, min(300, grid_size))
    small     = resize_for_processing(img, grid_size)

    print("  Clustering colors…")
    palette, label_grid, edge_density = quantize_image(small, n_colors)

    print("  Smoothing…")
    label_grid = smooth_label_grid_adaptive(label_grid, edge_density, passes=smooth)
    label_grid = flood_fill_small_regions(label_grid, min_size=30)

    print("  Rendering…")
    pbn     = build_pbn_image(label_grid, palette, scale=scale)
    pal_fig = build_palette_legend(palette)

    return pbn, pal_fig, palette


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image",    nargs="?", default=None)
    parser.add_argument("--colors", type=int,  default=14)
    parser.add_argument("--size",   type=int,  default=None)
    parser.add_argument("--scale",  type=int,  default=8)
    parser.add_argument("--smooth", type=int,  default=1)
    parser.add_argument("--output", type=str,  default=None)
    args = parser.parse_args()

    print("\n🎨  Paint by Number Generator")
    print("─" * 42)

    img, img_path = load_image(args.image)
    output_stem   = args.output or (img_path.stem + "_pbn")

    print(f"  Image  : {img_path.name}  ({img.width}×{img.height})")
    print(f"  Colors : {args.colors}  |  Scale : {args.scale}×")

    pbn, pal_fig, palette = run(
        img,
        n_colors   = max(4, min(30, args.colors)),
        scale      = max(3, min(12, args.scale)),
        smooth     = args.smooth,
        grid_size  = args.size,
    )

    pbn.save(f"{output_stem}_pbn.png")
    pal_fig.savefig(f"{output_stem}_palette.png", dpi=150,
                    bbox_inches="tight", facecolor="#f8f8f8")

    # Side-by-side guide
    fig = plt.figure(figsize=(16, 8), facecolor="#f4f4f4")
    gs  = GridSpec(1, 2, figure=fig, wspace=0.04)
    ax1 = fig.add_subplot(gs[0]); ax1.imshow(img);  ax1.axis("off")
    ax1.set_title("Original", fontsize=12, fontweight="bold")
    ax2 = fig.add_subplot(gs[1]); ax2.imshow(pbn); ax2.axis("off")
    ax2.set_title("Paint by Number", fontsize=12, fontweight="bold")
    patches = [mpatches.Patch(color=[v/255 for v in palette[i]],
               label=f"{i+1} {color_name(*palette[i])} {rgb_to_hex(*palette[i])}")
               for i in range(len(palette))]
    ax2.legend(handles=patches, loc="lower center",
               bbox_to_anchor=(0.5, -0.03 - 0.03 * len(palette)),
               ncol=min(4, len(palette)), fontsize=7.5, framealpha=0.92)
    plt.savefig(f"{output_stem}_guide.png", dpi=150,
                bbox_inches="tight", facecolor="#f4f4f4")
    plt.show()

    print(f"\n  Done ✓  —  saved: {output_stem}_pbn/palette/guide.png\n")


if __name__ == "__main__":
    main()
