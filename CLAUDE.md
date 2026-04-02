# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run dev server
streamlit run app.py

# Run via launcher (mimics packaged behavior — auto-opens browser)
python launcher.py

# Build distributable (requires pyinstaller)
pyinstaller --noconfirm --onedir --name PrintCalC --add-data "app.py:." launcher.py
```

## Architecture

Single-file Streamlit app (`app.py`) with no backend server or database. Everything runs locally in the browser session.

**Data flow:**
1. User uploads a PDF → `analyze_pdf_pages()` renders each page via PyMuPDF at the chosen DPI, converts pixels to numpy arrays, and classifies each pixel as white/color/black using RGB threshold + CMY chroma detection
2. Raw per-page stats (white%, color%, black%) feed into `compute_pricing()` which looks up the applicable price slab
3. Results are displayed as a dataframe with metrics summary

**Pricing model:**
- Two independent pricing profiles stored in `st.session_state`: `single_*` and `duplex_*`
- Each profile has two slab tables: color slabs (keyed by color%) and B/W slabs (keyed by non-white%)
- Printer selection ("Color printer" vs "B/W printer") determines which slab table is used at pricing time
- Blank pages (white ≥ 99.999%) always cost ₹0
- Color printer with 0% color content falls back to the lowest color slab price

**Packaging:**
- `launcher.py` wraps Streamlit in a subprocess, finds a free port, and opens the browser — used as the PyInstaller entry point
- CI builds distributable ZIPs for Windows and macOS via `.github/workflows/build.yml`; releases are handled by `.github/workflows/release.yml`
- PyInstaller bundles `app.py` as a data file alongside the launcher executable
