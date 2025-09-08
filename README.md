# PrintCalC 🖨️ — Fair per-page print pricing (MVP)

## Downloads
- **macOS**: [PrintCalC-mac.zip](https://github.com/Carnage725/PrintCalC/releases/latest/download/PrintCalC-mac.zip)
- **Windows**: coming soon


**PrintCalC** analyzes each PDF page locally and gives a simple, shop-friendly price:
- Choose **printer** for the whole job: **Color** or **B/W**
- Two pricing profiles: **Single-sided** and **Duplex** (each has its own slabs)
- Per-side measurements: **White %**, **Color %**, **Black %**
- **Blank pages are free**
- Border ignore (mm), **Fast (200dpi)** / **Accurate (300dpi)** modes
- No accounts, no server — runs on your machine

> This repo contains the Streamlit MVP and a tiny launcher for packaging.

---

## Quick start (dev)

```bash
# create & activate a venv (recommended)
python -m venv .venv

# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# run
streamlit run app.py
