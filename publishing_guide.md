# Publishing Guide — Road Vehicle Analytics

## Deployment Options

| Platform | Cost | Best For | GPU Support |
|---|---|---|---|
| **Streamlit Community Cloud** | Free | Demo & sharing | ❌ No |
| **Hugging Face Spaces** | Free | ML apps with GPU | ✅ Yes (paid) |
| **Railway / Render** | Free tier | Full control | ❌ No |
| **Your Own Server** | Varies | Production use | ✅ Yes |

> **Recommended:** Streamlit Community Cloud — free, easiest, directly supports Streamlit apps.

---

## Option 1 — Streamlit Community Cloud (Recommended)

### Prerequisites
- GitHub account with your code pushed (see `github_upload_guide.md`)
- Streamlit Community Cloud account (free)

### Step 1: Prepare Your Repo

Your repo needs these files at the root (you already have them):
```
streamlit_app.py        ← Streamlit will auto-detect this
requirements.txt        ← Dependencies
```

Add a `.streamlit/config.toml` for dark theme (optional but looks better):

```toml
[theme]
base = "dark"
primaryColor = "#38bdf8"

[server]
maxUploadSize = 500
```

### Step 2: Create `packages.txt` for System Dependencies

OpenCV needs system libraries. Create this file at the project root:

```
libgl1-mesa-glx
libglib2.0-0
```

### Step 3: Sign Up for Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **Sign up** → **Continue with GitHub**
3. Authorize Streamlit to access your GitHub repos

### Step 4: Deploy

1. Click **New app**
2. Fill in:
   - **Repository:** `your-username/road-vehicle-analytics`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
3. Click **Deploy!**

### Step 5: Wait for Build

- First deploy takes **5–10 minutes** (installs all dependencies)
- You'll get a public URL like: `https://your-app-name.streamlit.app`

### ⚠️ Limitations on Streamlit Cloud

| Limitation | Impact |
|---|---|
| **No GPU** | YOLOv8 inference will be very slow (CPU only) |
| **1 GB RAM limit** (free tier) | Large videos may crash |
| **No persistent storage** | Uploaded videos and results disappear on reboot |
| **YOLO weights not in repo** | Need to auto-download on first run |

### Fix: Auto-Download YOLO Weights

Since `.pt` files are gitignored, add this to the top of `streamlit_app.py` (after imports):

```python
# Auto-download YOLO weights if not present
from pathlib import Path
for model in ["yolov8n.pt", "yolov8m.pt"]:
    if not Path(model).exists():
        try:
            from ultralytics import YOLO
            YOLO(model)  # Downloads automatically
        except Exception:
            pass
```

### Your Published URL
```
https://your-app-name.streamlit.app
```
Share this link with anyone — they can use the dashboard without installing anything.

---

## Option 2 — Hugging Face Spaces

Better for ML apps — supports larger files and optional GPU.

### Step 1: Create a Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **Create new Space**
3. Choose:
   - **SDK:** Streamlit
   - **Hardware:** CPU Basic (free) or GPU (paid)
4. Clone the space repo locally

### Step 2: Copy Your Files

```bash
# Clone your HF space
git clone https://huggingface.co/spaces/YOUR_USERNAME/road-vehicle-analytics
cd road-vehicle-analytics

# Copy project files
cp -r /path/to/Final-Project/{streamlit_app.py,requirements.txt,src,configs} .
```

### Step 3: Add `app.py` Wrapper

Hugging Face expects `app.py`, so create a simple wrapper:

```python
# app.py — HuggingFace wrapper
import subprocess
subprocess.run(["streamlit", "run", "streamlit_app.py", "--server.port", "7860"])
```

### Step 4: Push

```bash
git add -A
git commit -m "Deploy Road Vehicle Analytics"
git push
```

Your app will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/road-vehicle-analytics`

---

## Option 3 — Self-Hosted (Your Own Server / VPS)

For full control, GPU support, and no limits.

### Step 1: Set Up Server

Any Linux server works (AWS EC2, DigitalOcean, your college server, etc.)

```bash
# SSH into your server
ssh user@your-server-ip

# Install Python 3.10+
sudo apt update && sudo apt install python3.10 python3.10-venv python3-pip -y

# Install OpenCV dependencies
sudo apt install libgl1-mesa-glx libglib2.0-0 -y
```

### Step 2: Clone and Setup

```bash
git clone https://github.com/YOUR_USERNAME/road-vehicle-analytics.git
cd road-vehicle-analytics

python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[vision,dashboard]"
```

### Step 3: Copy YOLO Weights

```bash
# From your local machine:
scp yolov8m.pt yolov8n.pt user@your-server-ip:~/road-vehicle-analytics/
```

### Step 4: Run with Public Access

```bash
streamlit run streamlit_app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true
```

### Step 5: Keep Running (Background)

```bash
# Using nohup
nohup streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &

# OR using screen
screen -S dashboard
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
# Press Ctrl+A, then D to detach
```

### Step 6: (Optional) Add Domain + HTTPS with Nginx

```nginx
# /etc/nginx/sites-available/analytics
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/analytics /etc/nginx/sites-enabled/
sudo certbot --nginx -d your-domain.com  # Free HTTPS
sudo systemctl restart nginx
```

---

## Quick Comparison

| | Streamlit Cloud | Hugging Face | Self-Hosted |
|---|---|---|---|
| **Setup time** | 5 min | 10 min | 30 min |
| **Cost** | Free | Free (CPU) | $5-20/mo |
| **GPU** | ❌ | ✅ (paid) | ✅ (if available) |
| **Custom domain** | ❌ | ❌ | ✅ |
| **Large video uploads** | Limited | Limited | Unlimited |
| **Best for** | Sharing/demo | ML showcase | Production |

---

## Recommended Path for College Project

1. **Push to GitHub** first (follow `github_upload_guide.md`)
2. **Deploy on Streamlit Cloud** for a shareable demo link
3. **Include the URL in your PPT** (Slide 15) so evaluators can try it live
4. **Run locally for the actual demo** — faster and more reliable than cloud
