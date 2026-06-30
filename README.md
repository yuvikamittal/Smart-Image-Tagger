# Smart Image Tagger

A small web app that sends an image (upload or URL) to **Azure AI Vision** and
displays the caption, tags, confidence scores, and OCR text it returns.

The frontend never talks to Azure directly — it calls a small Flask backend,
which holds the Azure Vision key as a server-side environment variable. This
is the piece that was missing in the original single-file HTML demo (it asked
you to paste the key into the page, which would have exposed it to anyone
opening dev tools).

## Repo structure

```
smart-image-tagger/
├── app.py                # Flask backend — proxies requests to Azure AI Vision
├── requirements.txt       # Python dependencies
├── .gitignore
├── README.md
└── static/
    └── index.html         # Frontend UI (served by Flask at "/")
```

Everything Azure App Service needs is at the repo root: `app.py` and
`requirements.txt`. Keep it this flat — Azure's Linux Python build (Oryx)
looks for `requirements.txt` in the root and runs `pip install` automatically.

## How it works

```
Browser  →  POST /analyze  →  Flask (app.py)  →  Azure AI Vision  →  JSON back to browser
```

`VISION_KEY` and `VISION_ENDPOINT` live only in App Service's environment —
they're read with `os.environ.get(...)` in `app.py` and never appear in any
file you commit or any response sent to the browser.

## 1. Local development

```bash
git clone https://github.com/<your-username>/smart-image-tagger.git
cd smart-image-tagger
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt

export VISION_KEY="your-azure-vision-key"
export VISION_ENDPOINT="https://your-resource.cognitiveservices.azure.com"

python app.py
```

Visit `http://localhost:5000`.

## 2. Create the Azure resources

1. In the [Azure Portal](https://portal.azure.com), create a **Computer
   Vision** (Azure AI Vision) resource. Copy **Key 1** and the **Endpoint**
   from its "Keys and Endpoint" page.
2. Create an **App Service** (Linux, Python 3.11+ runtime, F1 free tier is
   fine to start).

## 3. Configure App Settings (this replaces the old config UI)

In the App Service resource → **Settings → Environment variables** (or
**Configuration → Application settings** on older portal UI), add:

| Name              | Value                                                      |
|-------------------|-------------------------------------------------------------|
| `VISION_KEY`      | your Computer Vision key                                    |
| `VISION_ENDPOINT` | `https://your-resource.cognitiveservices.azure.com`          |

Save, then restart the app. This is the step you mentioned — the endpoint and
key go here, not in the HTML/JS.

## 4. Set the startup command

App Service → **Settings → Configuration → General settings → Startup
Command**:

```
gunicorn --bind=0.0.0.0 --timeout 600 app:app
```

(`gunicorn` is already in `requirements.txt`.)

## 5. Deploy

### Option A — GitHub Actions via Deployment Center (recommended)

1. Push this repo to GitHub.
2. In App Service → **Deployment → Deployment Center**, choose **GitHub** as
   the source, authorize, and pick your repo/branch.
3. Azure generates a GitHub Actions workflow (`.github/workflows/*.yml`) and
   commits it to your repo automatically. Every push to the branch
   redeploys.

### Option B — Azure CLI

```bash
az login
az webapp up \
  --name <your-app-name> \
  --resource-group <your-resource-group> \
  --runtime "PYTHON:3.11" \
  --sku F1
```

Run this from inside the repo folder. For subsequent deploys:

```bash
az webapp deploy --resource-group <your-resource-group> --name <your-app-name> --src-path .
```

### Option C — VS Code Azure App Service extension

Right-click the folder → **Deploy to Web App**, pick your subscription and
App Service.

## 6. Verify

- `https://<your-app-name>.azurewebsites.net/` → should load the UI.
- `https://<your-app-name>.azurewebsites.net/health` → should return
  `{"status": "ok", "configured": true}`. If `configured` is `false`, the App
  Settings weren't saved/restarted correctly.

## Notes

- `/analyze` accepts either `{"url": "..."}` or `{"image_base64": "..."}` —
  matches the two ways the frontend can supply an image (URL field or
  drag-and-drop/file upload).
- Because the frontend and backend are served from the same origin, there's
  no CORS configuration to worry about.
- If you later split the frontend onto a separate static host (e.g. Azure
  Static Web Apps), you'll need to add CORS headers in `app.py` and update
  the `fetch('/analyze')` call to a full URL.
