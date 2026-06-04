# From Scratch

### 1. Create the GCP VM.

- Machine type: `g2-standard-4`
- GPU: `1 x NVIDIA L4`
- OS: Ubuntu 22.04
- Boot disk: at least 100 GB SSD
- Allow HTTP and HTTPS traffic.

### 2. SSH into the VM.

The console can be opened via the Activate Cloud Shell button in the browser.

### 3. Update Ubuntu.

```bash
sudo apt update && sudo apt upgrade -y
```

### 4. Install required system packages.

```bash
sudo apt install -y \
    nginx \
    git \
    python3-pip \
    python3-venv \
    build-essential \
    pciutils
```

### 5. Install NVIDIA GPU drivers.

```bash
curl -L \
https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz \
--output cuda_installer.pyz
```

```bash
sudo python3 cuda_installer.pyz install_driver
```

```bash
sudo reboot
```

Reconnect via SSH after reboot.

```bash
nvidia-smi
```

You should see the NVIDIA L4 GPU listed.

### 6. Install Ollama.

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 7. Pull models.

```bash
ollama pull llama3.2
ollama pull llama3.2-vision
```

### 8. Clone the repo.

```bash
cd /opt

sudo mkdir -p /opt/recipe-rag

sudo chown -R $USER:$USER /opt/recipe-rag
```

```bash
git clone https://github.com/farzanashaju/recipe-rag /opt/recipe-rag
```

### 9. Create a virtual environment.

```bash
cd /opt/recipe-rag

python3 -m venv venv

source venv/bin/activate
```

### 10. Install dependencies.

```bash
pip install --upgrade pip

pip install -r requirements.txt
```

### 11. Data setup.

The repository intentionally excludes generated datasets and vector databases from version control.

After cloning the repository, you must generate the local dataset and embeddings manually.

```bash
playwright install chromium
```

```bash
python scripts/scrape-swiggy-recipes.py
python scripts/build-rag.py
```

### 12. Install systemd service.

```bash
sudo cp deploy/recipe-rag.service \
/etc/systemd/system/recipe-rag.service
```

```bash
sudo systemctl daemon-reload

sudo systemctl enable recipe-rag

sudo systemctl start recipe-rag

sudo systemctl status recipe-rag
```

### 13. Enable nginx.

```bash
sudo cp deploy/recipe-rag.nginx \
/etc/nginx/sites-available/recipe-rag
```

```bash
sudo ln -s \
/etc/nginx/sites-available/recipe-rag \
/etc/nginx/sites-enabled/
```

```bash
sudo rm /etc/nginx/sites-enabled/default

sudo nginx -t

sudo systemctl restart nginx
```

### 14. Access the website!

Visit `http://YOUR_EXTERNAL_IP`.

# Resume VM

```bash
sudo systemctl start ollama

sudo systemctl start recipe-rag

sudo systemctl start nginx

sudo systemctl status recipe-rag nginx ollama
```

---

# Cloud Run Deployment (Alternative & Recommended)

Since this project contains a `Dockerfile` preconfigured for GCP Cloud Run, you can deploy it as a serverless container. This is much simpler than managing a VM.

### 1. Open Google Cloud Shell
Go to the [Google Cloud Console](https://console.cloud.google.com) and click **Activate Cloud Shell** (the `>_` terminal icon in the top right header).

### 2. Clone the Repository in Cloud Shell
Run the following commands in the Cloud Shell terminal:
```bash
git clone https://github.com/koushi19/recipe-rag.git
cd recipe-rag
```

### 3. Deploy to Cloud Run
Run the single-line deployment command. It automatically builds the container using Cloud Build, uploads it to Artifact Registry, and deploys it:
```bash
gcloud run deploy recipe-rag \
    --source . \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated
```

*Note: If prompted to enable the Artifact Registry, Cloud Build, or Cloud Run APIs, press `y` to enable them.*

### 4. Get the URL
Once the deployment finishes, the terminal will display the Service URL:
```
Service URL: https://recipe-rag-xxxxxx-uc.a.run.app
```
Copy this URL and use it in your **FreshAuditor** app configuration as the backend endpoint!

