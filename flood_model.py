"""Flood mapping with a U-Net trained on Sen1Floods11 (Sentinel-1 VV/VH in dB).

Everything the app needs: loading the model, reading and checking GeoTIFFs,
predicting with overlapping tiles (no seams at tile edges), optional
test-time-augmentation uncertainty, and change detection.
"""
import io
import json
import math

import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
from rasterio.io import MemoryFile
from rasterio.warp import Resampling, reproject

MAX_PIXELS = 16_000_000   # ~40 x 40 km at 10 m; larger uploads are rejected to keep CPU time reasonable


# ----------------------------------------------------------------------------- model
class FloodModel:
    def __init__(self, weights_path, config_path, device="cpu"):
        with open(config_path) as f:
            self.cfg = json.load(f)
        self.device = device
        self.model = smp.Unet(self.cfg.get("encoder", "resnet34"), encoder_weights=None,
                              in_channels=self.cfg.get("in_channels", 3), classes=1)
        state = torch.load(weights_path, map_location="cpu")
        state = {k: (v.float() if v.is_floating_point() else v) for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.eval().to(device)
        self.mean = np.asarray(self.cfg["mean"], np.float32)[:, None, None]
        self.std = np.asarray(self.cfg["std"], np.float32)[:, None, None]
        self.lo, self.hi = self.cfg.get("clip_db", [-50, 1])
        self.nodata = self.cfg.get("nodata_db", -50)
        self.tile = self.cfg.get("tile", 512)
        self.threshold = self.cfg.get("threshold", 0.5)
        self.unc_threshold = self.cfg.get("unc_threshold", 0.05)

    def preprocess(self, s1_db):
        """s1_db: (2, H, W) VV, VH in dB -> normalised (3, H, W): VV, VH, VV-VH (same as training)."""
        s1 = np.nan_to_num(s1_db.astype(np.float32), nan=self.nodata)
        s1 = np.clip(s1, self.lo, self.hi)
        x = np.concatenate([s1, (s1[0] - s1[1])[None]], axis=0)
        return ((x - self.mean) / self.std).astype(np.float32)

    @torch.no_grad()
    def predict(self, s1_db, tta_views=0, overlap=128, batch_size=4, progress=None):
        """Water probability for a whole scene.

        Tiles overlap and are blended with a smooth window, so there are no seams
        at tile edges. With tta_views (4 or 8) each tile is also predicted in rotated/
        flipped versions; the spread between them is returned as uncertainty.
        Returns prob (H, W), unc (H, W) or None, valid (H, W).
        """
        valid = np.isfinite(s1_db[0]) & (s1_db[0] > self.nodata)
        x = self.preprocess(s1_db)
        _, H, W = x.shape
        t, stride, m = self.tile, self.tile - overlap, overlap // 2

        # pad so tiles cover the image exactly; reflection gives border pixels some context
        Hp, Wp = H + 2 * m, W + 2 * m
        extra_h = (t - Hp) if Hp < t else (-(Hp - t)) % stride
        extra_w = (t - Wp) if Wp < t else (-(Wp - t)) % stride
        xp = np.pad(x, ((0, 0), (m, m + extra_h), (m, m + extra_w)), mode="reflect")
        _, Hp, Wp = xp.shape

        h = 0.05 + 0.95 * np.hanning(t).astype(np.float32)
        wwin = np.outer(h, h)
        acc_p = np.zeros((Hp, Wp), np.float32)
        acc_u = np.zeros((Hp, Wp), np.float32) if tta_views else None
        acc_w = np.zeros((Hp, Wp), np.float32)

        coords = [(r, c) for r in range(0, Hp - t + 1, stride) for c in range(0, Wp - t + 1, stride)]
        views = {0: [(0, False)], 4: [(k, False) for k in range(4)],
                 8: [(k, f) for k in range(4) for f in (False, True)]}[tta_views]

        for i in range(0, len(coords), batch_size):
            chunk = coords[i:i + batch_size]
            xb = torch.from_numpy(np.stack([xp[:, r:r + t, c:c + t] for r, c in chunk])).to(self.device)
            probs = []
            for k, flip in views:
                v = torch.rot90(xb, k, (2, 3))
                if flip:
                    v = torch.flip(v, (3,))
                p = torch.sigmoid(self.model(v))
                if flip:
                    p = torch.flip(p, (3,))
                probs.append(torch.rot90(p, -k, (2, 3)))
            probs = torch.stack(probs)[:, :, 0].cpu().numpy()      # (views, B, t, t)
            mean = probs.mean(0)
            std = probs.std(0, ddof=1) if len(views) > 1 else None
            for j, (r, c) in enumerate(chunk):
                acc_p[r:r + t, c:c + t] += mean[j] * wwin
                acc_w[r:r + t, c:c + t] += wwin
                if std is not None:
                    acc_u[r:r + t, c:c + t] += std[j] * wwin
            if progress:
                progress(min(1.0, (i + len(chunk)) / len(coords)))

        prob = (acc_p / acc_w)[m:m + H, m:m + W]
        unc = (acc_u / acc_w)[m:m + H, m:m + W] if tta_views else None
        prob[~valid] = 0
        if unc is not None:
            unc[~valid] = 0
        return prob, unc, valid

    def n_tiles(self, H, W, overlap=128):
        stride = self.tile - overlap
        f = lambda n: max(1, math.ceil((n + overlap - self.tile) / stride) + 1)
        return f(H) * f(W)


# ----------------------------------------------------------------------------- rasters
class InputError(ValueError):
    """Raised with a user-facing explanation when an uploaded file can't be used."""


def read_s1(data):
    """Read a Sentinel-1 GeoTIFF (bytes or path). Returns (s1_db (2,H,W), profile, notes)."""
    notes = []
    try:
        mf = MemoryFile(data) if isinstance(data, (bytes, bytearray)) else None
        ds = mf.open() if mf else rasterio.open(data)
    except Exception as e:
        raise InputError("This file could not be read as a GeoTIFF. Export it as a .tif with the "
                         "script in tools/export_sentinel1_gee.js.") from e
    with ds:
        if ds.count < 2:
            raise InputError(f"The image has {ds.count} band, but the model needs two: VV and VH.")
        if ds.width * ds.height > MAX_PIXELS:
            raise InputError(f"The image is {ds.width} x {ds.height} pixels. The limit is about 40 x 40 km "
                             f"at 10 m ({MAX_PIXELS / 1e6:.0f} million pixels); crop it to a smaller area.")
        if ds.crs is None:
            raise InputError("The image has no map projection, so it can't be placed on a map or measured.")
        desc = [(d or "").upper() for d in ds.descriptions]
        if "VV" in desc and "VH" in desc:
            idx = [desc.index("VV") + 1, desc.index("VH") + 1]
        else:
            idx = [1, 2]
            notes.append("Band names not found; assuming band 1 = VV and band 2 = VH.")
        s1 = ds.read(idx).astype(np.float32)
        if ds.nodata is not None:
            s1[s1 == ds.nodata] = np.nan
        profile = ds.profile.copy()

    s1[~np.isfinite(s1)] = np.nan
    finite = s1[0][np.isfinite(s1[0])]
    if finite.size == 0:
        raise InputError("The image contains no valid pixels.")
    sample = finite[:: max(1, finite.size // 200_000)]
    if np.median(sample) > 0 and np.percentile(sample, 99) < 20:
        s1 = 10 * np.log10(np.clip(s1, 1e-6, None))
        notes.append("Values looked like linear backscatter, so they were converted to decibels.")
    elif np.median(sample) > 5:
        raise InputError("Pixel values don't look like Sentinel-1 backscatter in decibels (land is "
                         "usually around -5 to -15 dB). Check that this is a VV/VH radar image.")

    px = pixel_size_m(profile)
    if not 7 <= px <= 15:
        notes.append(f"Pixel size is about {px:.0f} m; the model was trained at 10 m, so results may be less reliable.")
    return s1, profile, notes


def pixel_size_m(profile):
    t, crs = profile["transform"], profile["crs"]
    if crs.is_projected:
        return math.sqrt(abs(t.a * t.e))
    lat = t.f + t.e * profile["height"] / 2
    return math.sqrt(abs(t.a * 111_320 * math.cos(math.radians(lat)) * t.e * 110_540))


def pixel_area_ha(profile):
    return pixel_size_m(profile) ** 2 / 10_000


def align_to(s1_ref, ref_profile, target_profile):
    """Reproject a reference image onto the flood image's grid if they differ."""
    same = (ref_profile["crs"] == target_profile["crs"] and ref_profile["transform"] == target_profile["transform"]
            and ref_profile["width"] == target_profile["width"] and ref_profile["height"] == target_profile["height"])
    if same:
        return s1_ref, False
    out = np.full((2, target_profile["height"], target_profile["width"]), np.nan, np.float32)
    for b in range(2):
        reproject(s1_ref[b], out[b], src_transform=ref_profile["transform"], src_crs=ref_profile["crs"],
                  dst_transform=target_profile["transform"], dst_crs=target_profile["crs"],
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear)
    return out, True


def change_detection(prob_flood, valid_flood, prob_ref, valid_ref, threshold=0.5):
    """Flooded = water in the flood image, not water in the reference image."""
    water_flood = (prob_flood > threshold) & valid_flood
    water_ref = (prob_ref > threshold) & valid_ref
    flooded = water_flood & ~water_ref & valid_ref
    return water_flood, water_ref, flooded


def to_geotiff(bands, names, profile):
    """Write float32 bands to an in-memory GeoTIFF and return its bytes."""
    p = {k: profile[k] for k in ("crs", "transform", "width", "height")}
    p.update(driver="GTiff", count=len(bands), dtype="float32", compress="lzw", nodata=None)
    with MemoryFile() as mf:
        with mf.open(**p) as ds:
            for i, (b, n) in enumerate(zip(bands, names), start=1):
                ds.write(b.astype(np.float32), i)
                ds.set_band_description(i, n)
        return mf.read()
