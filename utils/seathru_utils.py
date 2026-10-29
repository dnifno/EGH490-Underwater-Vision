"""
Sea-thru (Akkaynak & Treibitz, CVPR 2019) underwater image enhancement for monocular video frames, using Depth Anything V2 to supply the range map.

Usage:
    from seathru_utils import load_seathru, enhance_frame
    depth_bundle = load_seathru(device)
    enhanced = enhance_frame(pil_image, depth_bundle, device)

The Sea-thru algorithm code below (backscatter estimation, illumination estimation, wideband attenuation, image recovery, white balance) is vendored
from the community Python port https://github.com/hainh/sea-thru (MIT License, Copyright (c) 2020 John Gibson). 
That port breaks on current numpy / Pillow / scikit-image, so it is copied here with these changes only:

  1. Removed matplotlib.use('TkAgg'), rawpy, argparse and plotting code.
  2. construct_neighborhood_map / refine_neighborhood_map REPLACED with a vectorised equivalent (scipy.ndimage). The original is a pure-Python
     flood fill from random start pixels and is far too slow for video. The result is deterministic but not pixel-identical to the original.
  3. refine_wideband_attentuation: added a fallback to the linear model if every curve_fit restart fails (original would crash on coefs=None).
  4. Added a seed so runs are repeatable (curve_fit restarts use np.random).

Everything else (formulas, defaults) is unchanged. The depth source differs from the paper --> used SfM range maps in metres. 
Here the range map is RELATIVE monocular depth rescaled to an assumed metric span (see disparity_to_range).
"""

import math
import sys

import numpy as np
import scipy.optimize
import scipy.stats
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import closing, disk
try:  # skimage >= 0.25 deprecates square()
    from skimage.morphology import footprint_rectangle as _rect
    def square(n):
        return _rect((n, n))
except ImportError:  # older skimage
    from skimage.morphology import square
from skimage.restoration import denoise_bilateral

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"  # Apache-2.0


# ---------------------------------------------------------------------------
# Vendored from hainh/sea-thru (MIT)
# ---------------------------------------------------------------------------

def find_backscatter_estimation_points(img, depths, num_bins=10, fraction=0.01, max_vals=20, min_depth_percent=0.0):
    """Darkest RGB triplets in each depth bin are taken as backscatter samples."""
    z_max, z_min = np.max(depths), np.min(depths)
    min_depth = z_min + (min_depth_percent * (z_max - z_min))
    z_ranges = np.linspace(z_min, z_max, num_bins + 1)
    img_norms = np.mean(img, axis=2)
    points_r, points_g, points_b = [], [], []
    for i in range(len(z_ranges) - 1):
        a, b = z_ranges[i], z_ranges[i + 1]
        locs = np.where(np.logical_and(depths > min_depth, np.logical_and(depths >= a, depths <= b)))
        norms_in_range, px_in_range, depths_in_range = img_norms[locs], img[locs], depths[locs]
        arr = sorted(zip(norms_in_range, px_in_range, depths_in_range), key=lambda x: x[0])
        points = arr[:min(math.ceil(fraction * len(arr)), max_vals)]
        points_r.extend([(z, p[0]) for n, p, z in points])
        points_g.extend([(z, p[1]) for n, p, z in points])
        points_b.extend([(z, p[2]) for n, p, z in points])
    return np.array(points_r), np.array(points_g), np.array(points_b)


def find_backscatter_values(B_pts, depths, restarts=10, max_mean_loss_fraction=0.1):
    """Fit B(z) = B_inf(1-exp(-beta_B z)) + J'exp(-beta_D' z); fall back to a line."""
    B_vals, B_depths = B_pts[:, 1], B_pts[:, 0]
    z_max, z_min = np.max(depths), np.min(depths)
    max_mean_loss = max_mean_loss_fraction * (z_max - z_min)
    coefs = None
    best_loss = np.inf

    def estimate(depths, B_inf, beta_B, J_prime, beta_D_prime):
        return (B_inf * (1 - np.exp(-1 * beta_B * depths))) + (J_prime * np.exp(-1 * beta_D_prime * depths))

    def loss(B_inf, beta_B, J_prime, beta_D_prime):
        return np.mean(np.abs(B_vals - estimate(B_depths, B_inf, beta_B, J_prime, beta_D_prime)))

    bounds_lower = [0, 0, 0, 0]
    bounds_upper = [1, 5, 1, 5]
    for _ in range(restarts):
        try:
            optp, _pcov = scipy.optimize.curve_fit(
                f=estimate,
                xdata=B_depths,
                ydata=B_vals,
                p0=np.random.random(4) * bounds_upper,
                bounds=(bounds_lower, bounds_upper),
            )
            l = loss(*optp)
            if l < best_loss:
                best_loss = l
                coefs = optp
        except RuntimeError as re:
            print(re, file=sys.stderr)
    if coefs is None or best_loss > max_mean_loss:
        slope, intercept, _r, _p, _se = scipy.stats.linregress(B_depths, B_vals)
        BD = (slope * depths) + intercept
        return BD, np.array([slope, intercept])
    return estimate(depths, *coefs), coefs


def estimate_illumination(img, B, neighborhood_map, num_neighborhoods, p=0.5, f=2.0, max_iters=100, tol=1E-5):
    """Illumination map from local colour-space averaging within depth neighbourhoods."""
    D = img - B
    avg_cs = np.zeros_like(img)
    avg_cs_prime = np.copy(avg_cs)
    sizes = np.zeros(num_neighborhoods)
    locs_list = [None] * num_neighborhoods
    for label in range(1, num_neighborhoods + 1):
        locs_list[label - 1] = np.where(neighborhood_map == label)
        sizes[label - 1] = np.size(locs_list[label - 1][0])
    for _ in range(max_iters):
        for label in range(1, num_neighborhoods + 1):
            locs = locs_list[label - 1]
            size = sizes[label - 1] - 1
            avg_cs_prime[locs] = (1 / size) * (np.sum(avg_cs[locs]) - avg_cs[locs])
        new_avg_cs = (D * p) + (avg_cs_prime * (1 - p))
        if np.max(np.abs(avg_cs - new_avg_cs)) < tol:
            break
        avg_cs = new_avg_cs
    return f * denoise_bilateral(np.maximum(0, avg_cs))


def estimate_wideband_attentuation(depths, illum, radius=6, max_val=10.0):
    eps = 1E-8
    BD = np.minimum(max_val, -np.log(illum + eps) / (np.maximum(0, depths) + eps))
    mask = np.where(np.logical_and(depths > eps, illum > eps), 1, 0)
    refined_attenuations = denoise_bilateral(closing(np.maximum(0, BD * mask), disk(radius)))
    return refined_attenuations, []


def calculate_beta_D(depths, a, b, c, d):
    return (a * np.exp(b * depths)) + (c * np.exp(d * depths))


def filter_data(X, Y, radius_fraction=0.01):
    idxs = np.argsort(X)
    X_s = X[idxs]
    Y_s = Y[idxs]
    x_max, x_min = np.max(X), np.min(X)
    radius = (radius_fraction * (x_max - x_min))
    ds = np.cumsum(X_s - np.roll(X_s, (1,)))
    dX = [X_s[0]]
    dY = [Y_s[0]]
    tempX, tempY = [], []
    pos = 0
    for i in range(1, ds.shape[0]):
        if ds[i] - ds[pos] >= radius:
            tempX.append(X_s[i])
            tempY.append(Y_s[i])
            idxs = np.argsort(tempY)
            med_idx = len(idxs) // 2
            dX.append(tempX[med_idx])
            dY.append(tempY[med_idx])
            pos = i
        else:
            tempX.append(X_s[i])
            tempY.append(Y_s[i])
    return np.array(dX), np.array(dY)


def refine_wideband_attentuation(depths, illum, estimation, restarts=10, min_depth_fraction=0.1,
                                 max_mean_loss_fraction=np.inf, l=1.0, radius_fraction=0.01):
    """Fit the two-term exponential beta_D(z) to the per-pixel attenuation estimates."""
    eps = 1E-8
    z_max, z_min = np.max(depths), np.min(depths)
    min_depth = z_min + (min_depth_fraction * (z_max - z_min))
    max_mean_loss = max_mean_loss_fraction * (z_max - z_min)
    coefs = None
    best_loss = np.inf
    locs = np.where(np.logical_and(illum > 0, np.logical_and(depths > min_depth, estimation > eps)))

    def calculate_reconstructed_depths(depths, illum, a, b, c, d):
        eps = 1E-5
        return -np.log(illum + eps) / (calculate_beta_D(depths, a, b, c, d) + eps)

    def loss(a, b, c, d):
        return np.mean(np.abs(depths[locs] - calculate_reconstructed_depths(depths[locs], illum[locs], a, b, c, d)))

    dX, dY = filter_data(depths[locs], estimation[locs], radius_fraction)
    for _ in range(restarts):
        try:
            optp, _pcov = scipy.optimize.curve_fit(
                f=calculate_beta_D,
                xdata=dX,
                ydata=dY,
                p0=np.abs(np.random.random(4)) * np.array([1., -1., 1., -1.]),
                bounds=([0, -100, 0, -100], [100, 0, 100, 0]))
            L = loss(*optp)
            if L < best_loss:
                best_loss = L
                coefs = optp
        except RuntimeError as re:
            print(re, file=sys.stderr)
    if coefs is None or best_loss > max_mean_loss:  # (coefs is None check added; see module docstring)
        slope, intercept, _r, _p, _se = scipy.stats.linregress(depths[locs], estimation[locs])
        BD = (slope * depths + intercept)
        return l * BD, np.array([slope, intercept])
    BD = l * calculate_beta_D(depths, *coefs)
    return BD, coefs


def wbalance_no_red_10p(img):
    """White balance from the brightest 10% of green and blue (red is unreliable underwater)."""
    n = int(round(-1 * np.size(img[:, :, 0]) * 0.1))
    dg = 1.0 / np.mean(np.sort(img[:, :, 1], axis=None)[n:])
    db = 1.0 / np.mean(np.sort(img[:, :, 2], axis=None)[n:])
    dsum = dg + db
    dg = dg / dsum * 2.
    db = db / dsum * 2.
    img[:, :, 0] *= (db + dg) / 2
    img[:, :, 1] *= dg
    img[:, :, 2] *= db
    return img


def scale(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def recover_image(img, depths, B, beta_D, nmap):
    """Remove backscatter, undo attenuation, then white balance."""
    res = (img - B) * np.exp(beta_D * np.expand_dims(depths, axis=2))
    res = np.maximum(0.0, np.minimum(1.0, res))
    res[nmap == 0] = 0
    res = scale(wbalance_no_red_10p(res))
    res[nmap == 0] = img[nmap == 0]
    return res


# ---------------------------------------------------------------------------
# Replacement for construct_/refine_neighborhood_map (vectorised; see docstring)
# ---------------------------------------------------------------------------

def construct_neighborhood_map(depths, epsilon=0.1, min_size=50, radius=3):
    """
    Partition the image into connected regions of similar range.

    Original: flood fill from random start pixels, accepting pixels within
    +/- epsilon*range of the start pixel's depth, then merging regions smaller
    than min_size into the nearest neighbour and applying a closing.
    Here: bin depth with width 2*epsilon*range (same acceptance window), label
    connected components per bin, merge small regions into the nearest large
    one with a distance transform, then the same closing. Returns
    (nmap, n_neighborhoods) with labels contiguous from 1.
    """
    z_min, z_max = float(np.min(depths)), float(np.max(depths))
    width = max(2.0 * epsilon * (z_max - z_min), 1e-6)
    bins = np.floor((depths - z_min) / width).astype(np.int32)

    nmap = np.zeros(depths.shape, dtype=np.int32)
    n = 0
    for b in np.unique(bins):
        lab, k = ndi.label(bins == b)
        m = lab > 0
        nmap[m] = lab[m] + n
        n += k

    sizes = np.bincount(nmap.ravel(), minlength=n + 1)
    keep = sizes >= min_size
    keep[0] = False
    if not keep.any():  # degenerate (e.g. flat depth): one neighbourhood
        return np.ones(depths.shape, dtype=np.int32), 1

    remap = np.zeros(n + 1, dtype=np.int32)
    remap[keep] = np.arange(1, keep.sum() + 1, dtype=np.int32)
    refined = remap[nmap]

    idx = ndi.distance_transform_edt(refined == 0, return_distances=False, return_indices=True)
    refined = refined[idx[0], idx[1]]
    refined = closing(refined, square(radius))

    uniq, inv = np.unique(refined, return_inverse=True)  # keep labels contiguous
    refined = (inv.reshape(refined.shape) + (1 if uniq[0] != 0 else 0)).astype(np.int32)
    return refined, int(refined.max())


# ---------------------------------------------------------------------------
# Depth: Depth Anything V2 (relative inverse depth) -> assumed metric range
# ---------------------------------------------------------------------------

def load_seathru(device, model_id=DEPTH_MODEL_ID):
    """Load the monocular depth model. Returns (image_processor, depth_model)."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


def estimate_relative_disparity(pil_image, depth_bundle, device):
    """(H, W) float32 relative inverse depth at the image's resolution. LARGER = CLOSER."""
    import torch
    import torch.nn.functional as F
    processor, model = depth_bundle
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        pred = model(**inputs).predicted_depth  # (1, h, w)
    w, h = pil_image.size
    pred = F.interpolate(pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False)
    return pred.squeeze().float().cpu().numpy()


def disparity_to_range(disp, min_range_m=0.5, range_span_m=2.0, clip_percentiles=(2, 98)):
    """
    Map relative disparity to an ASSUMED metric camera-to-scene range.

    Same mapping as the port's preprocess_monodepth_depth_map
    (range = span * (1 - normalised_disparity) + min_range), except the
    normalisation uses percentiles instead of min/max so a few outlier pixels
    don't set the scale. Nearest pixels -> min_range_m, farthest ->
    min_range_m + range_span_m. These two numbers are guesses about the scene,
    not measurements: they are the main thing to tune.
    """
    lo, hi = np.percentile(disp, clip_percentiles)
    d = np.clip((disp - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)
    return range_span_m * (1.0 - d) + min_range_m


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1 / 2.4)) - 0.055)


def enhance_frame(pil_image, depth_bundle=None, device=None, depth_map=None,
                  min_range_m=0.5, range_span_m=2.0, max_size=640,
                  f=2.0, l=0.5, p=0.01, min_depth=0.1, spread_data_fraction=0.01,
                  linearise=False, seed=0, return_debug=False):
    """
    Run Sea-thru on one RGB PIL frame and return an enhanced RGB PIL image at the original resolution.

    depth_bundle : from load_seathru(); used when depth_map is None.
    depth_map : optional (H, W) range in metres at ANY resolution (e.g. real altitude/stereo/sonar-derived). Skips monocular estimation.
    min_range_m, range_span_m : assumed range of nearest pixel and the span to the farthest pixel (monocular case only). TUNE THESE.
    max_size : the pipeline runs on a copy with its long side limited to this (it is slow); the result is resized back up, so fine detail is lost. Raising it improves detail at the cost of runtime.
    f, l, p, min_depth, spread_data_fraction : port defaults, unchanged (f brightness, l attenuation balance, p illuminant locality).
    linearise : undo/redo the sRGB curve around the algorithm. The port does not do this (it feeds gamma-encoded 8-bit); off by default.
    return_debug : also return dict with 'depth' (working-res range map),'nmap', 'B' and 'beta_D'.
    """
    np.random.seed(seed)
    pil_image = pil_image.convert("RGB")
    orig_size = pil_image.size

    work = pil_image.copy()
    work.thumbnail((max_size, max_size), Image.LANCZOS)

    if depth_map is None:
        if depth_bundle is None or device is None:
            raise ValueError("Provide depth_map, or depth_bundle and device.")
        disp = estimate_relative_disparity(work, depth_bundle, device)
        depths = disparity_to_range(disp, min_range_m, range_span_m)
    else:
        depths = np.array(Image.fromarray(np.asarray(depth_map, dtype=np.float32)).resize(work.size, Image.BILINEAR))
        depths = depths.astype(np.float32)

    img = np.float32(work) / 255.0
    if linearise:
        img = _srgb_to_linear(img).astype(np.float32)

    ptsR, ptsG, ptsB = find_backscatter_estimation_points(img, depths, fraction=0.01, min_depth_percent=min_depth)
    Br, _ = find_backscatter_values(ptsR, depths, restarts=25)
    Bg, _ = find_backscatter_values(ptsG, depths, restarts=25)
    Bb, _ = find_backscatter_values(ptsB, depths, restarts=25)

    nmap, n = construct_neighborhood_map(depths, 0.1, min_size=50)

    illR = estimate_illumination(img[:, :, 0], Br, nmap, n, p=p, max_iters=100, tol=1E-5, f=f)
    illG = estimate_illumination(img[:, :, 1], Bg, nmap, n, p=p, max_iters=100, tol=1E-5, f=f)
    illB = estimate_illumination(img[:, :, 2], Bb, nmap, n, p=p, max_iters=100, tol=1E-5, f=f)

    bdr, _ = estimate_wideband_attentuation(depths, illR)
    bdr, _ = refine_wideband_attentuation(depths, illR, bdr, radius_fraction=spread_data_fraction, l=l)
    bdg, _ = estimate_wideband_attentuation(depths, illG)
    bdg, _ = refine_wideband_attentuation(depths, illG, bdg, radius_fraction=spread_data_fraction, l=l)
    bdb, _ = estimate_wideband_attentuation(depths, illB)
    bdb, _ = refine_wideband_attentuation(depths, illB, bdb, radius_fraction=spread_data_fraction, l=l)

    B = np.stack([Br, Bg, Bb], axis=2)
    beta_D = np.stack([bdr, bdg, bdb], axis=2)
    recovered = recover_image(img, depths, B, beta_D, nmap)

    if linearise:
        recovered = _linear_to_srgb(recovered)

    out = Image.fromarray((np.clip(recovered, 0, 1) * 255).round().astype(np.uint8))
    if out.size != orig_size:
        out = out.resize(orig_size, Image.LANCZOS)

    if return_debug:
        return out, {"depth": depths, "nmap": nmap, "B": B, "beta_D": beta_D}
    return out