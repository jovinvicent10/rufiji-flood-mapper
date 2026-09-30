---
title: Rufiji Flood Mapper
emoji: 🌊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Map floods in Sentinel-1 radar images with deep learning
---

# Rufiji flood mapper

A web app that finds floodwater in **Sentinel-1 radar satellite images** using a U-Net deep learning model.
Upload a radar image of a flood (and, ideally, one of the same area in a normal season) and the app maps which
land was newly flooded, how many hectares that is, and where the model is unsure.

It comes from an Advanced Machine Learning mini-project on detecting floods as rare events, with a case study of
the **April 2024 Rufiji River floods in Tanzania**.

**Live app:** `https://huggingface.co/spaces/YOUR-HF-USERNAME/rufiji-flood-mapper`
**Model:** `https://huggingface.co/YOUR-HF-USERNAME/rufiji-flood-unet`

## What it does

1. Reads your Sentinel-1 GeoTIFF (VV and VH bands in decibels, 10 m pixels) and checks it is usable.
2. Runs the U-Net over the image in overlapping 512 × 512 tiles, blended so there are no seams.
3. If you add a normal-season image, keeps only **new** water: water in the flood image that was not there before.
4. Optionally flags uncertain areas by comparing the model's answers on rotated and flipped copies of each tile.
5. Shows the result on a satellite basemap, with the flooded area in hectares, and lets you download a GeoTIFF,
   a PNG figure and a CSV summary.

## Getting images for your own area

The model only works on **Sentinel-1 radar images**. Phone photos, drone photos and ordinary satellite pictures
will not work. The easiest way to get the right images, free, for any area:

1. Sign up for [Google Earth Engine](https://earthengine.google.com) (free for non-commercial use).
2. Open the [Code Editor](https://code.earthengine.google.com) and paste in
   [`tools/export_sentinel1_gee.js`](tools/export_sentinel1_gee.js).
3. Change the area, the flood dates and the normal-season dates at the top, then select **Run**.
4. In the **Tasks** tab, run both exports. Two GeoTIFFs appear in your Google Drive.
5. Download them and upload them in the app: `S1_flood.tif` first, `S1_reference.tif` second.

Keep the area under about 40 × 40 km. On the free CPU hardware, a 15 × 15 km pair takes about one to two
minutes without the uncertainty map.

## Model and results

| | |
|---|---|
| Architecture | U-Net, ImageNet-pretrained ResNet-34 encoder |
| Input | VV, VH and VV − VH (dB), normalised with training statistics |
| Training data | Sen1Floods11 hand-labelled chips (Bonafilia et al., 2020): 252 train, 89 validation, 90 test |
| Test IoU / precision / recall | 0.678 / 0.839 / 0.780 |
| Unseen flood (Bolivia) IoU | 0.613 |
| Rufiji, April 2024 | about 35,900 ha newly flooded; 80% mapped with high confidence |

Explainable AI in the notebook showed the model relies most on the **VH** channel, not just on dark VV pixels, and
that its uncertainty points to its errors: flagged pixels were wrong about 7.6 times more often than the rest.

## Limitations

- **Towns:** flooded streets can look bright on radar, not dark, so urban flooding is often missed.
- **Vegetation:** water under trees or mangroves is hard to see.
- **Look-alikes:** smooth tarmac, dry sand and radar shadows behind hills can look like water.
- **Timing:** Sentinel-1 passes about every 12 days, so short floods can be missed.
- **Validation:** results in Tanzania have not been checked against ground truth. Treat them as estimates.

## Project structure

```
app.py                      Streamlit web app
flood_model.py              model loading, tiled prediction, uncertainty, change detection, GeoTIFF I/O
requirements.txt            Python packages (CPU build of PyTorch)
Dockerfile                  container used by Hugging Face Spaces
.streamlit/config.toml      theme and upload limit
tools/export_sentinel1_gee.js     get input images from Google Earth Engine
tools/colab_export_model.py       package the trained model from the Colab notebook
.github/workflows/sync_to_hf.yml  copy each GitHub push to the Hugging Face Space
notebook/                   the training and analysis notebook
```

## Run it on your own computer

```bash
git clone https://github.com/YOUR-GITHUB-USERNAME/rufiji-flood-mapper.git
cd rufiji-flood-mapper
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export FLOOD_MODEL_REPO=YOUR-HF-USERNAME/rufiji-flood-unet  # Windows: set FLOOD_MODEL_REPO=...
streamlit run app.py
```

Or put `model.pt`, `config.json` and the `demo` folder in `models/` to run without downloading.

See [DEPLOY.md](DEPLOY.md) to publish your own copy.

## Credits

- Data: Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020). *Sen1Floods11: a georeferenced dataset to
  train and test deep learning flood algorithms for Sentinel-1.* CVPR Workshops.
- Sentinel-1 imagery: Copernicus programme, European Space Agency, via Google Earth Engine.
- Permanent water: JRC Global Surface Water. Land cover: ESA WorldCover.
- Author: [Your name], [Institution].
