# PrintCalC

A local-first PDF print pricing tool that analyzes ink coverage per page and calculates fair, shop-ready costs — no cloud, no accounts, no data leaving your machine.

**[Download for macOS / Windows →](../../releases)**

---

## What it does

Print shops typically charge flat rates that disadvantage customers with lightly-inked pages. PrintCalC solves this by measuring actual ink coverage and pricing each page against configurable cost slabs — the same way a fair shop would.

- Renders every PDF page via PyMuPDF and classifies each pixel as **white**, **color**, or **black/grey** using RGB threshold + CMY chroma analysis
- Supports **Color** and **B/W** printer modes with separate pricing profiles for **single-sided** and **duplex** jobs
- Blank pages (100% white) are always free
- Border-ignore margin strips header/footer noise before measurement
- Fast (200 DPI) and Accurate (300 DPI) analysis modes
- All processing is local — the PDF never leaves the device

## Tech stack

| Layer | Choice |
|---|---|
| UI | Streamlit |
| PDF rendering | PyMuPDF (fitz) |
| Pixel analysis | NumPy |
| Data / display | Pandas |
| Packaging | PyInstaller (onedir) |
| CI/CD | GitHub Actions (build + release) |

## How it works

1. **PDF → pixels** — each page is rasterized at the chosen DPI using PyMuPDF's `get_pixmap`, then loaded into a NumPy array
2. **Pixel classification** — white pixels are identified by RGB threshold; among the rest, per-pixel CMY chroma (`max(C,M,Y) - min(C,M,Y)`) determines color vs grey
3. **Slab lookup** — the resulting coverage percentages are matched against user-editable price slabs (stored in Streamlit session state, with overlap/gap validation)
4. **Output** — a per-page breakdown table plus total pages, sheets, price, and processing time

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
app.py          # entire application — analysis, pricing logic, Streamlit UI
launcher.py     # PyInstaller entry point: spawns Streamlit, opens browser
requirements.txt
.github/
  workflows/
    build.yml   # builds Windows + macOS ZIPs on push to main
    release.yml # attaches ZIPs to GitHub Releases on tag push
```

## Key design decisions

- **Single-file app** — all logic lives in `app.py` to keep the packaged artifact simple and auditable
- **No backend** — Streamlit's session state is the only persistence; there's nothing to deploy or secure
- **Configurable slabs** — shop owners can adjust pricing tiers directly in the UI without touching code; slabs are validated for overlaps and gaps before use
- **Launcher pattern** — `launcher.py` wraps Streamlit in a subprocess so PyInstaller can produce a native executable that behaves like a desktop app
