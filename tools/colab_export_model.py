# =============================================================================
# Paste into a new cell at the end of the notebook and run it in the SAME Colab session as the notebook, after Sections 1-15.
# It packages everything the web app needs:
#   model.pt       best U-Net weights (half precision, about 49 MB)
#   config.json    normalisation values, thresholds and test scores
#   demo/*.tif     a 15 x 15 km Rufiji example (flood + dry season) for the app's example mode
#   README.md      a short model card
# then saves the package to Google Drive and uploads it to the Hugging Face Hub.
# =============================================================================
import json, os, shutil
import numpy as np, torch, rasterio
from rasterio.transform import from_origin

EXPORT = "/content/flood_app_export"
os.makedirs(f"{EXPORT}/demo", exist_ok=True)

# ---- 1. Model weights (half precision halves the size; the app converts back) --------------------
best_name = max(models, key=lambda n: results[n]["test"]["IoU"])
state = {k: (v.half() if v.is_floating_point() else v) for k, v in models[best_name].state_dict().items()}
torch.save(state, f"{EXPORT}/model.pt")

# ---- 2. Config: everything needed to reproduce the preprocessing exactly --------------------------
config = {
    "model_name": best_name,
    "architecture": "Unet", "encoder": "resnet34", "in_channels": 3,
    "channels": ["VV", "VH", "VV-VH"],
    "mean": [float(v) for v in MEAN.ravel()], "std": [float(v) for v in STD.ravel()],
    "clip_db": [-50, 1], "nodata_db": -50, "tile": 512, "pixel_size_m": 10,
    "threshold": 0.5,
    "unc_threshold": float(UNC_THR) if "UNC_THR" in globals() else 0.05,
    "test_metrics": {k: float(v) for k, v in results[best_name]["test"].items()},
    "bolivia_metrics": {k: float(v) for k, v in results[best_name]["bolivia"].items()},
    "training_data": "Sen1Floods11 hand-labelled chips (252 train, 89 validation, 90 test)",
}
json.dump(config, open(f"{EXPORT}/config.json", "w"), indent=2)
print("Model:", best_name, "| test IoU", round(config["test_metrics"]["IoU"], 3))

# ---- 3. Example images: the 1536 x 1536 px window (15 x 15 km) with the most flooding --------------
if "S1_flood" in globals() and "FLOODED" in globals():
    N = 1536
    best_rc, best_sum = (0, 0), -1
    for r0 in range(0, FLOODED.shape[0] - N + 1, 256):
        for c0 in range(0, FLOODED.shape[1] - N + 1, 256):
            s = FLOODED[r0:r0 + N, c0:c0 + N].sum()
            if s > best_sum:
                best_rc, best_sum = (r0, c0), s
    r0, c0 = best_rc
    tfm = from_origin(x_min + c0 * SCALE, y_max - r0 * SCALE, SCALE, SCALE)
    for name, arr in [("rufiji_flood_2024-04", S1_flood), ("rufiji_dry_2023", S1_dry)]:
        with rasterio.open(f"{EXPORT}/demo/{name}.tif", "w", driver="GTiff", height=N, width=N, count=2,
                           dtype="float32", crs=CRS, transform=tfm, nodata=-50, compress="lzw", predictor=3) as f:
            f.write(arr[:, r0:r0 + N, c0:c0 + N].astype(np.float32))
            f.set_band_description(1, "VV"); f.set_band_description(2, "VH")
    print(f"Example window: rows {r0}-{r0 + N}, cols {c0}-{c0 + N}, {best_sum * SCALE**2 / 1e4:,.0f} ha flooded in the notebook")
else:
    print("S1_flood not in memory - run Section 14 first if you want the example images.")

# ---- 4. Model card ---------------------------------------------------------------------------------
tm = config["test_metrics"]
open(f"{EXPORT}/README.md", "w").write(f"""---
license: mit
tags: [flood-mapping, sentinel-1, sar, semantic-segmentation, tanzania]
---
# Rufiji flood U-Net

U-Net (ResNet-34 encoder) that maps surface water in Sentinel-1 radar images. Used by the Rufiji flood mapper app.

- **Input:** Sentinel-1 IW GRD, VV and VH in dB, 10 m pixels; the model adds VV-VH as a third channel.
- **Training data:** {config['training_data']}.
- **Test scores (threshold 0.5):** IoU {tm['IoU']:.3f}, precision {tm['Precision']:.3f}, recall {tm['Recall']:.3f}.
- **Files:** `model.pt` (state dict, float16), `config.json` (normalisation and thresholds), `demo/` (Rufiji, April 2024).
- **Limitations:** misses many urban floods (radar double bounce), water under vegetation; can confuse smooth
  surfaces, sand and radar shadow with water. Not validated with ground truth in Tanzania.
""")

# ---- 5. Save to Google Drive -----------------------------------------------------------------------
from google.colab import drive
drive.mount("/content/drive")
shutil.copytree(EXPORT, "/content/drive/MyDrive/flood_app_export", dirs_exist_ok=True)
print("Saved to Google Drive: My Drive/flood_app_export")

# ---- 6. Upload to the Hugging Face Hub -------------------------------------------------------------
# Create a free account at huggingface.co, then Settings > Access Tokens > New token (type: Write).
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
from huggingface_hub import login, create_repo, upload_folder
login()                                             # paste your WRITE token when asked
REPO_ID = "YOUR-HF-USERNAME/rufiji-flood-unet"      # <-- change YOUR-HF-USERNAME
create_repo(REPO_ID, repo_type="model", exist_ok=True)
upload_folder(folder_path=EXPORT, repo_id=REPO_ID, repo_type="model")
print(f"Uploaded: https://huggingface.co/{REPO_ID}")
