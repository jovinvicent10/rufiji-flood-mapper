# Deploying the Rufiji flood mapper

Everything here is free. You need three accounts: Google (you already use Colab), GitHub, and Hugging Face.
The model file lives on the Hugging Face Hub, the code lives on GitHub, and the app runs on a Hugging Face Space.

```
Colab notebook ──(export)──> Hugging Face Hub: model.pt, config.json, demo images
GitHub repo: app code ──(automatic sync)──> Hugging Face Space: the running app ──> downloads the model
```

## Step 1: Export the model from Colab

1. Create a free account at [huggingface.co](https://huggingface.co/join).
2. In Hugging Face, go to **Settings → Access Tokens → Create new token**, choose **Write**, and copy it.
3. Open your notebook in Colab. The trained models must be in memory, so either continue a session where you ran
   Sections 1–15, or select **Runtime → Run all** first.
4. Add a new cell at the end, paste in `tools/colab_export_model.py`, and change `YOUR-HF-USERNAME` to your
   Hugging Face username.
5. Run the cell. Allow Google Drive access when asked, then paste your token when `login()` asks for it.
6. When it finishes, open `https://huggingface.co/YOUR-HF-USERNAME/rufiji-flood-unet`. You should see
   `model.pt`, `config.json`, `README.md` and a `demo` folder.

## Step 2: Put the code on GitHub

1. Create a free account at [github.com](https://github.com/signup).
2. Select **New repository**, name it `rufiji-flood-mapper`, make it **Public**, and don't add a README
   (this project already has one).
3. Upload the files. The simplest way: on the empty repository page, select **uploading an existing file**, drag in
   everything from this folder **including the hidden `.streamlit` and `.github` folders**, and select
   **Commit changes**. (On Mac, press Cmd+Shift+. in Finder to show hidden folders; on Windows, enable
   View → Hidden items.)
4. Also create a `notebook` folder and upload your `.ipynb` there, so people can see how the model was trained.
5. In `README.md` and `app.py`, replace `YOUR-HF-USERNAME`, `YOUR-GITHUB-USERNAME`, `[Your name]` and
   `[Institution]`. You can edit files directly on GitHub with the pencil icon.

## Step 3: Create the Hugging Face Space

1. On Hugging Face, select **New → Space**.
2. Name it `rufiji-flood-mapper`, choose **Docker** as the SDK and the **Blank** template, hardware
   **CPU basic (free)**, visibility **Public**, then **Create Space**.
   (Hugging Face no longer offers Streamlit as a built-in SDK; the included `Dockerfile` runs Streamlit instead.)
3. In the Space, go to **Settings → Variables and secrets → New variable**:
   name `FLOOD_MODEL_REPO`, value `YOUR-HF-USERNAME/rufiji-flood-unet`.

## Step 4: Connect GitHub to the Space

1. In your GitHub repository, go to **Settings → Secrets and variables → Actions**.
2. Under **Secrets**, add `HF_TOKEN` with your Hugging Face write token.
3. Under **Variables**, add `HF_SPACE` with `YOUR-HF-USERNAME/rufiji-flood-mapper`.
4. Go to the **Actions** tab, select **Sync to Hugging Face Space → Run workflow**.
5. Open your Space. It shows **Building** for about 5–10 minutes the first time, then **Running**.

From now on, every change you commit on GitHub is copied to the Space automatically, and the app rebuilds.

## Step 5: Test it

1. Open the Space, choose **Example: Rufiji floods, April 2024**, and select **Map floods**.
   The first run downloads the model (a few seconds) and takes about a minute on the free CPU.
2. Check the numbers look sensible and the map sits on the Rufiji River.
3. Share the Space link. Anyone can use it without an account.

## If something goes wrong

| Problem | What to do |
|---|---|
| "The model could not be loaded" | Check the `FLOOD_MODEL_REPO` variable in the Space settings matches the model repo name exactly, and that the model repo is public. |
| Space build fails | Open the **Logs** tab in the Space. Most failures are a typo in `requirements.txt` or a missing file. |
| GitHub Action fails with "authentication" | The `HF_TOKEN` secret must be a **Write** token. |
| Uploads fail with a 403 error | Make sure the `Dockerfile` still includes `--server.enableXsrfProtection=false`. |
| The app is slow | Turn off the uncertainty map, or use a smaller area. The free Space has 2 CPU cores. |
| The Space is "Sleeping" | Free Spaces sleep after about 48 hours without visitors; opening the link wakes it in a minute or two. |
