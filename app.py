"""Rufiji flood mapper: find floodwater in Sentinel-1 radar images with a U-Net."""
import hashlib
import io
import os

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from rasterio.warp import transform_bounds
from streamlit_folium import st_folium

from flood_model import (FloodModel, InputError, align_to, change_detection, pixel_area_ha,
                         read_s1, to_geotiff)

MODEL_REPO = os.environ.get("FLOOD_MODEL_REPO", "YOUR-HF-USERNAME/rufiji-flood-unet")
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "models")
DEMO_FILES = {"flood": "demo/rufiji_flood_2024-04.tif", "ref": "demo/rufiji_dry_2023.tif"}

# Map colours, the same as in the notebook
RED, YELLOW, BLUE = (214, 40, 30), (255, 206, 0), (30, 96, 214)

st.set_page_config(page_title="Rufiji flood mapper", page_icon="🌊", layout="wide")
torch.set_num_threads(max(1, os.cpu_count() or 1))


# ----------------------------------------------------------------------------- loading
def _fetch(filename):
    """Use a local copy in ./models if present, otherwise download from the Hugging Face Hub."""
    local = os.path.join(LOCAL_DIR, filename)
    if os.path.exists(local):
        return local
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=MODEL_REPO, filename=filename)


@st.cache_resource(show_spinner="Loading the model…")
def load_model():
    return FloodModel(_fetch("model.pt"), _fetch("config.json"))


@st.cache_data(show_spinner=False)
def load_bytes(filename):
    with open(_fetch(filename), "rb") as f:
        return f.read()


def run_model(model, s1, tta, label):
    bar = st.progress(0.0, text=f"Analysing the {label} image…")
    out = model.predict(s1, tta_views=tta, progress=lambda f: bar.progress(f, text=f"Analysing the {label} image… {f:.0%}"))
    bar.empty()
    return out


# ----------------------------------------------------------------------------- display helpers
def display_step(shape, max_side=1400):
    return max(1, int(np.ceil(max(shape) / max_side)))


def overlay_rgba(water, flooded, lowconf, step):
    """Coloured transparent layer: new flood red, flagged flood yellow, normal/other water blue."""
    w = water[::step, ::step]
    rgba = np.zeros(w.shape + (4,), np.uint8)
    if flooded is None:
        rgba[w] = BLUE + (200,)
    else:
        f, lc = flooded[::step, ::step], lowconf[::step, ::step]
        rgba[w & ~f] = BLUE + (200,)
        rgba[f] = RED + (200,)
        rgba[f & lc] = YELLOW + (220,)
    return rgba


def web_map(rgba, profile):  # rgba may be downsampled; bounds come from the full image
    west, south, east, north = transform_bounds(profile["crs"], "EPSG:4326",
                                                *rasterio_bounds(profile))
    m = folium.Map(location=[(south + north) / 2, (west + east) / 2], tiles=None, control_scale=True)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri, Maxar, Earthstar Geographics", name="Satellite").add_to(m)
    folium.TileLayer("OpenStreetMap", name="Streets").add_to(m)
    folium.raster_layers.ImageOverlay(rgba, bounds=[[south, west], [north, east]], name="Flood map",
                                      mercator_project=True).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[south, west], [north, east]])
    return m


def rasterio_bounds(profile):
    t, w, h = profile["transform"], profile["width"], profile["height"]
    xs, ys = [t.c, t.c + t.a * w], [t.f, t.f + t.e * h]
    return min(xs), min(ys), max(xs), max(ys)


def sar_panel(ax, vv, title, step):
    ax.imshow(vv[::step, ::step], cmap="gray", vmin=-25, vmax=0)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def figure_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ----------------------------------------------------------------------------- page
st.title("Rufiji flood mapper")
st.write("Find floodwater in Sentinel-1 radar images. A U-Net trained on 11 flood events worldwide marks "
         "water pixel by pixel; comparing a flood image with a normal-season image shows what was newly flooded.")

try:
    model = load_model()
except Exception as e:
    st.error(f"The model could not be loaded from `{MODEL_REPO}`. Set the FLOOD_MODEL_REPO variable to your "
             f"Hugging Face model repository, or put model.pt and config.json in the models folder.\n\n{e}")
    st.stop()

with st.sidebar:
    st.header("Images")
    source = st.radio("What would you like to analyse?", ["Example: Rufiji floods, April 2024", "My own Sentinel-1 images"])
    if source.startswith("My own"):
        up_flood = st.file_uploader("Flood-period image (GeoTIFF, VV and VH in dB)", type=["tif", "tiff"])
        up_ref = st.file_uploader("Normal-season image, optional but recommended", type=["tif", "tiff"],
                                  help="A dry-season image of the same area. Without it, the app shows all water, "
                                       "including rivers and lakes, rather than only new flooding.")
        st.caption("Need images? The README explains how to export them for any area with Google Earth Engine.")
    st.header("Options")
    tta_label = st.radio("Uncertainty map", ["Off (fastest)", "4 views", "8 views (as in the notebook)"],
                         help="The model looks at each tile in several rotated and flipped versions. Where the "
                              "answers disagree, the flood is flagged in yellow for checking. More views take longer.")
    tta = {"Off (fastest)": 0, "4 views": 4, "8 views (as in the notebook)": 8}[tta_label]
    run = st.button("Map floods", type="primary", width="stretch")

with st.expander("How the model was trained and where it struggles"):
    cfg = model.cfg
    tm = cfg.get("test_metrics", {})
    st.markdown(
        f"- **Model:** {cfg.get('model_name', 'U-Net')} (ResNet-34 encoder), trained on Sen1Floods11 hand-labelled chips.\n"
        f"- **Test scores:** IoU {tm.get('IoU', float('nan')):.3f}, precision {tm.get('Precision', float('nan')):.3f}, "
        f"recall {tm.get('Recall', float('nan')):.3f}.\n"
        "- **Input:** Sentinel-1 IW GRD, VV and VH bands in decibels, 10 m pixels (as provided by Google Earth Engine).\n"
        "- **Known limits:** flooded towns can look bright rather than dark on radar and are often missed; water under "
        "trees or mangroves is hard to see; smooth tarmac, sand and radar shadows can look like water.\n"
        "- **Results are estimates.** Check yellow areas and anything important on the ground before acting on them.")

if not run and "result" not in st.session_state:
    st.info("Choose the example or upload your images in the sidebar, then select **Map floods**.")
    st.stop()

# ----------------------------------------------------------------------------- run
if run:
    try:
        if source.startswith("Example"):
            flood_bytes, ref_bytes = load_bytes(DEMO_FILES["flood"]), load_bytes(DEMO_FILES["ref"])
            names = ("April 2024 flood", "dry season 2023")
        else:
            if up_flood is None:
                st.warning("Upload a flood-period image first.")
                st.stop()
            flood_bytes = up_flood.getvalue()
            ref_bytes = up_ref.getvalue() if up_ref else None
            names = ("flood-period", "normal-season")

        s1_f, prof, notes = read_s1(flood_bytes)
        s1_r = None
        if ref_bytes:
            s1_r, prof_r, notes_r = read_s1(ref_bytes)
            notes += [f"Normal-season image: {n}" for n in notes_r]
            s1_r, moved = align_to(s1_r, prof_r, prof)
            if np.isfinite(s1_r[0]).mean() < 0.1:
                raise InputError("The two images barely overlap. Export both for the same area.")
            if moved:
                notes.append("The normal-season image was reprojected onto the flood image's grid.")
    except InputError as e:
        st.error(str(e))
        st.stop()

    H, W = s1_f.shape[1:]
    n = model.n_tiles(H, W) * (2 if s1_r is not None else 1) * max(tta, 1)
    st.caption(f"{H} x {W} pixels, about {n} model passes.")

    key = hashlib.md5(flood_bytes[:1_000_000] + (ref_bytes or b"")[:1_000_000] + str((len(flood_bytes), tta)).encode()).hexdigest()
    if st.session_state.get("key") != key:
        p_f, u_f, v_f = run_model(model, s1_f, tta, names[0])
        res = dict(prof=prof, vv_f=s1_f[0], notes=notes, names=names, tta=tta, p_f=p_f, u_f=u_f, v_f=v_f)
        if s1_r is not None:
            p_r, u_r, v_r = run_model(model, s1_r, tta, names[1])
            res.update(vv_r=s1_r[0], p_r=p_r, u_r=u_r, v_r=v_r)
        st.session_state["result"], st.session_state["key"] = res, key

# ----------------------------------------------------------------------------- results
r = st.session_state["result"]
prof, thr = r["prof"], model.threshold
ha = pixel_area_ha(prof)
for note in r["notes"]:
    st.warning(note)

has_ref = "p_r" in r
if has_ref:
    water, water_ref, flooded = change_detection(r["p_f"], r["v_f"], r["p_r"], r["v_r"], thr)
    valid = r["v_f"] & r["v_r"]
else:
    water, flooded, valid = (r["p_f"] > thr) & r["v_f"], None, r["v_f"]

unc = None
if r["tta"]:
    unc = np.maximum(r["u_f"], r["u_r"]) if has_ref else r["u_f"]
target = flooded if has_ref else water
lowconf = (target & (unc >= model.unc_threshold)) if unc is not None else np.zeros_like(target)

cols = st.columns(4)
cols[0].metric("Area analysed", f"{valid.sum() * ha / 100:,.0f} km²")
cols[1].metric("Water in the flood image", f"{water.sum() * ha:,.0f} ha")
if has_ref:
    cols[2].metric("Newly flooded", f"{flooded.sum() * ha:,.0f} ha",
                   help="Water in the flood image that was not water in the normal-season image.")
if unc is not None:
    cols[3].metric("Flagged for checking", f"{100 * lowconf.sum() / max(target.sum(), 1):.1f}%",
                   help="Share of the mapped water or flood where the model's views disagree.")

if has_ref:
    legend = "Red: newly flooded. Yellow: flooding worth checking. Blue: water that is also there in the normal season."
    ov_flooded = flooded
else:
    legend = "Blue: water detected." + (" Yellow: water worth checking." if unc is not None else "")
    ov_flooded = (water & lowconf) if unc is not None else None   # flagged water drawn yellow
step = display_step(water.shape)
rgba = overlay_rgba(water, ov_flooded, lowconf, step)

tab_map, tab_img, tab_unc, tab_dl = st.tabs(["Map", "Radar images", "Uncertainty", "Download"])
with tab_map:
    st.markdown(legend)
    st_folium(web_map(rgba, prof), height=560, use_container_width=True, returned_objects=[])

with tab_img:
    ncol = 3 if has_ref else 2
    fig, ax = plt.subplots(1, ncol, figsize=(6 * ncol, 6))
    if has_ref:
        sar_panel(ax[0], r["vv_r"], f"Radar (VV), {r['names'][1]}", step)
    sar_panel(ax[ncol - 2], r["vv_f"], f"Radar (VV), {r['names'][0]}", step)
    base = np.clip((np.nan_to_num(r["vv_f"][::step, ::step], nan=0) + 25) / 25, 0, 1)
    img = np.dstack([base] * 3)
    mask = rgba[..., 3] > 0
    img[mask] = rgba[mask, :3] / 255
    ax[ncol - 1].imshow(img)
    ax[ncol - 1].set_title("Result", fontsize=11)
    ax[ncol - 1].axis("off")
    st.caption("Dark areas in the radar images are smooth surfaces, usually water.")
    png = figure_png(fig)
    st.image(png, width="stretch")

with tab_unc:
    if unc is None:
        st.info("Turn on the uncertainty map in the sidebar and select **Map floods** again.")
    else:
        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(unc[::step, ::step], cmap="magma", vmin=0, vmax=max(2 * model.unc_threshold, 1e-3))
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.03, label="Disagreement between views (brighter = less sure)")
        st.image(figure_png(fig), width="stretch")
        st.caption(f"Pixels above {model.unc_threshold:.3f} are flagged. On the test set, flagged pixels were wrong "
                   "about 7 times more often than the rest.")

with tab_dl:
    summary = {"Area analysed (km²)": round(valid.sum() * ha / 100, 1), "Water in flood image (ha)": round(water.sum() * ha)}
    if has_ref:
        summary["Newly flooded (ha)"] = round(flooded.sum() * ha)
    if unc is not None:
        summary["Flagged for checking (%)"] = round(100 * lowconf.sum() / max(target.sum(), 1), 1)
    st.dataframe(pd.DataFrame([summary]), hide_index=True, width="stretch")

    bands, names = [water.astype(np.float32), r["p_f"]], ["water", "water_probability"]
    if has_ref:
        bands.insert(0, flooded.astype(np.float32)); names.insert(0, "newly_flooded")
    if unc is not None:
        bands.append(unc); names.append("uncertainty")
    st.download_button("Download flood map (GeoTIFF)", to_geotiff(bands, names, prof),
                       file_name="flood_map.tif", mime="image/tiff", width="stretch")
    st.download_button("Download figure (PNG)", png, file_name="flood_map.png", mime="image/png",
                       width="stretch")
    st.download_button("Download summary (CSV)", pd.DataFrame([summary]).to_csv(index=False).encode(),
                       file_name="flood_summary.csv", mime="text/csv", width="stretch")
    st.caption("The GeoTIFF opens in QGIS or Google Earth Pro; bands: " + ", ".join(names) + ".")
