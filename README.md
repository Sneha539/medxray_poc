# D2 — Medical X‑ray Classifier POC (with Grad‑CAM + optional LLM explainer)

> **Educational demo only — not a medical device or diagnostic tool.**

This project fine‑tunes a pretrained vision model (DenseNet121 via `timm`) on a public chest X‑ray dataset (e.g., **Kaggle: Chest X‑Ray Images (Pneumonia)**).  
It serves a Flask UI where users can upload an X‑ray and get **diagnosis probabilities** with an **attention heatmap (Grad‑CAM)**.  
Optionally, it generates a plain‑language explanation using an **LLM** (OpenAI) if `OPENAI_API_KEY` is set. Without a key, it falls back to a safe templated explanation.

---

## Quickstart

### 0) Requirements
- Python 3.10+
- (Recommended) GPU with CUDA for training, but CPU will still work (slower).
- (Optional) OpenAI API key to enable natural‑language explanations.
```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt       
cp .env.example .env  # edit as needed
```
### 1) Get data
Use a small public dataset to start:

- **Kaggle: Chest X-Ray Images (Pneumonia)** — contains `train/`, `val/`, `test/` with `NORMAL/` and `PNEUMONIA/` folders.
  - Put the extracted folder anywhere, e.g. `data/chest_xray/` in this repo.

If your dataset does **not** already have `train/val/test`, use the helper:
```bash
python scripts/split_dataset.py --src data/raw_xrays --dst data/chest_xray --val 0.1 --test 0.1
```
### 2) Train
```bash    
python train.py --data_dir data/chest_xray --epochs 5 --batch_size 16 --img_size 224
```
Artifacts:
- `models/best_model.pt` — best weights by validation AUC
- `models/label_map.json` — class name ↔ index mapping
- `logs/train_log.csv` — metrics per epoch

### 3) Run the app
```bash
FLASK_APP=app/app.py flask run  # or: python app/app.py
```
Open http://127.0.0.1:5000 and upload an image.

### 4) Docker (optional)
```bash
docker build -t medxray_poc .
docker run -p 5000:5000 --env-file .env medxray_poc
```
---
## Project layout
```
medxray_poc/
├── app/
│   ├── app.py                 # Flask server
│   ├── templates/
│   │   └── index.html         # UI
│   └── static/
│       ├── css/styles.css
│       ├── overlays/          # Grad‑CAM images saved here
│       └── uploads/           # user uploads (auto‑cleaned)
├── models/                    # trained weights + label_map.json
├── utils/
│   ├── gradcam.py             # Grad‑CAM implementation
│   └── inference.py           # model load + preprocess + predict + heatmap
├── train.py                   # fine‑tune script
├── scripts/
│   └── split_dataset.py       # make train/val/test splits if needed
├── tests/
│   └── test_preprocess.py     # basic sanity/unit tests
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

---

## Notes & disclaimers
- **Not for clinical use.** Outputs are probabilistic and can be wrong.
- Respect dataset licenses.
- Never upload real PHI (protected health information) to third‑party servers.
- Keep your API keys private. The app reads keys from `.env` (dotenv).

---

## Common issues
- `CUDA out of memory`: lower `--batch_size`, or use CPU.
- Class imbalance: this repo uses **weighted BCE** automatically.
- Missing `best_model.pt`: train first, or the app will fallback to ImageNet‑pretrained weights (uninformative).

