# app.py
# 🖨️ PrintCalC — choose printer (Color or B/W) for the whole job
# Two pricing profiles: Single-sided & Duplex (used based on Sides selection)
# Blank pages (100% white) are free (₹0)
#
# Requirements:
#   pip install streamlit pymupdf numpy pandas

import time, math
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import streamlit as st

# ---------- Detection defaults ----------
DEFAULT_DPI_FAST = 200
DEFAULT_DPI_ACCURATE = 300
DEFAULT_WHITE_THRESH = 245
SENSITIVITY_PRESETS = {
    "Recommended": 0.05,
    "Lenient (more counted as B/W)": 0.08,
    "Strict (more counted as Color)": 0.03,
}

# ---------- Default slabs ----------
# Color slabs (you showed: 1–5: ₹8, 5–30: ₹10, 30–50: ₹14, 50–70: ₹18, 70–100: ₹22)
DEFAULT_COLOR_SLABS = pd.DataFrame(
    [
        {"min_pct":  1.0, "max_pct":  5.0,  "price":  8.0},
        {"min_pct":  5.0, "max_pct": 30.0,  "price": 10.0},
        {"min_pct": 30.0, "max_pct": 50.0,  "price": 14.0},
        {"min_pct": 50.0, "max_pct": 70.0,  "price": 18.0},
        {"min_pct": 70.0, "max_pct":100.0,  "price": 22.0},
    ]
)

# B/W slabs by non-white % (used only when "B/W printer" is selected)
DEFAULT_BW_SLABS_SINGLE = pd.DataFrame(
    [
        {"min_pct":  0.0, "max_pct": 40.0, "price": 3.0},
        {"min_pct": 40.0, "max_pct": 70.0, "price": 4.0},
        {"min_pct": 70.0, "max_pct":100.0, "price": 5.0},
    ]
)
DEFAULT_BW_SLABS_DUPLEX = pd.DataFrame(
    [
        {"min_pct":  0.0, "max_pct": 40.0, "price": 4.0},
        {"min_pct": 40.0, "max_pct": 70.0, "price": 5.0},
        {"min_pct": 70.0, "max_pct":100.0, "price": 6.0},
    ]
)

# ---------- Helpers ----------
def mm_to_px(mm: float, dpi: int) -> int:
    return int(round((mm / 25.4) * dpi))

def price_from_slabs(slabs_df: pd.DataFrame, pct: float):
    for _, row in slabs_df.iterrows():
        if row["min_pct"] <= pct <= row["max_pct"]:
            return float(row["price"])
    return None

def lowest_color_slab_price(slabs_df: pd.DataFrame) -> float:
    row = slabs_df.sort_values("min_pct").iloc[0]
    return float(row["price"])

def round_rupee(x: float) -> int:
    return int(round(x))

def analyze_pdf_pages(pdf_bytes: bytes, dpi: int, white_thresh: int,
                      chroma_thresh: float, ignore_border_mm: float):
    """
    Return list of dicts: {page, white_pct, color_pct, black_pct}
    - white%: R,G,B >= white_thresh
    - color%: among non-white, chroma > chroma_thresh
    - black%: remaining non-white pixels (grey/black)
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    results = []
    for idx, page in enumerate(doc, start=1):
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)

        h, w, _ = img.shape
        if ignore_border_mm > 0:
            margin = mm_to_px(ignore_border_mm, dpi)
            if margin * 2 < min(h, w):
                img = img[margin:h - margin, margin:w - margin, :]
                h, w, _ = img.shape

        total_px = h * w
        if total_px == 0:
            results.append({"page": idx, "white_pct": 0.0, "color_pct": 0.0, "black_pct": 0.0})
            continue

        r = img[:, :, 0].astype(np.uint16)
        g = img[:, :, 1].astype(np.uint16)
        b = img[:, :, 2].astype(np.uint16)
        white_mask = (r >= white_thresh) & (g >= white_thresh) & (b >= white_thresh)
        white_px = int(white_mask.sum())

        if white_px == total_px:
            results.append({"page": idx, "white_pct": 100.0, "color_pct": 0.0, "black_pct": 0.0})
            continue

        nw = ~white_mask
        rf = (r[nw].astype(np.float32)) / 255.0
        gf = (g[nw].astype(np.float32)) / 255.0
        bf = (b[nw].astype(np.float32)) / 255.0

        max_rgb = np.maximum.reduce([rf, gf, bf])
        k = 1.0 - max_rgb
        denom = np.maximum(1.0 - k, 1e-8)
        c = (1.0 - rf - k) / denom
        m = (1.0 - gf - k) / denom
        y = (1.0 - bf - k) / denom

        max_cmy = np.maximum.reduce([c, m, y])
        min_cmy = np.minimum.reduce([c, m, y])
        chroma = max_cmy - min_cmy
        color_mask = chroma > chroma_thresh

        color_px = int(color_mask.sum())
        gray_px = int((~color_mask).sum())

        results.append(
            {
                "page": idx,
                "white_pct": (white_px * 100.0) / total_px,
                "color_pct": (color_px * 100.0) / total_px,
                "black_pct": (gray_px * 100.0) / total_px,
            }
        )
    return results

def compute_pricing(rows, printer: str,
                    color_slabs: pd.DataFrame,
                    bw_slabs: pd.DataFrame,
                    round_per_side: bool = True):
    """
    printer: "Color printer" or "B/W printer"
    - Color printer: use Color slabs by color%, but if color%=0 and page not blank,
      charge the lowest Color slab. Blank pages (white=100%) cost ₹0.
    - B/W printer: use B/W slabs by non-white% (100 - white%). Blank pages ₹0.
    """
    out = []
    for r in rows:
        page = r["page"]
        white_pct = r["white_pct"]
        color_pct = r["color_pct"]
        black_pct = r["black_pct"]

        if white_pct >= 99.999:  # fully blank
            final = 0.0
            priced_by = "Blank"
            slab_price = 0.0
        elif printer == "Color printer":
            # Use color slabs; if 0% color (pure B/W content), charge lowest color slab
            base = price_from_slabs(color_slabs, color_pct)
            if base is None:
                base = lowest_color_slab_price(color_slabs)
            slab_price = base
            final = base
            priced_by = "Color slabs"
        else:
            # B/W printer: price by non-white %
            bw_ink_pct = 100.0 - white_pct
            base = price_from_slabs(bw_slabs, bw_ink_pct)
            if base is None:
                # Shouldn't happen if slabs cover 0..100, but fall back to last row's price
                base = float(bw_slabs.sort_values("max_pct").iloc[-1]["price"])
            slab_price = base
            final = base
            priced_by = "B/W slabs"

        final_price = round_rupee(final) if round_per_side else final
        out.append(
            {
                "page": page,
                "white_pct": white_pct,
                "color_pct": color_pct,
                "black_pct": black_pct,
                "priced_by": priced_by,
                "slab_price": slab_price,
                "price": final_price,
                "bw_ink_pct": (100.0 - white_pct) if printer == "B/W printer" else None,
            }
        )

    return pd.DataFrame(out).sort_values("page").reset_index(drop=True)

def ensure_slabs(df: pd.DataFrame, label: str):
    problems = []
    for i, row in df.iterrows():
        a, b = float(row["min_pct"]), float(row["max_pct"])
        if a > b:
            problems.append(f"{label} row {i+1}: min_pct > max_pct")
        if a < 0 or b > 100:
            problems.append(f"{label} row {i+1}: values must be within 0–100")
    df_sorted = df.sort_values(["min_pct", "max_pct"]).reset_index(drop=True)
    for i in range(1, len(df_sorted)):
        prev, cur = df_sorted.loc[i - 1], df_sorted.loc[i]
        if float(cur["min_pct"]) < float(prev["max_pct"]):
            problems.append(
                f"{label} rows {i}–{i+1} overlap "
                f"({prev['min_pct']}-{prev['max_pct']} vs {cur['min_pct']}-{cur['max_pct']})"
            )
    return problems, df_sorted

# ---------- UI ----------
st.set_page_config(page_title="PrintCalC — Fair Print Pricing", page_icon="🖨️", layout="wide")
st.title("🖨️ PrintCalC (PDF)")

# Sidebar
with st.sidebar:
    st.subheader("Printer & Sides")
    printer = st.radio("Choose printer", ["Color printer", "B/W printer"], index=0)
    sides = st.radio("Sides", ["Single-sided", "Back-to-back (duplex)"], index=0)

    st.subheader("Analysis")
    mode = st.radio("Mode", ["Fast", "Accurate"], index=0, help="Fast = 200 DPI, Accurate = 300 DPI")
    dpi = DEFAULT_DPI_FAST if mode == "Fast" else DEFAULT_DPI_ACCURATE

    st.subheader("Ignore page borders")
    ignore_mm = st.number_input("Ignore border (mm)", min_value=0.0, max_value=20.0, value=0.0, step=0.5)

    with st.expander("Advanced detection"):
        sensitivity_label = st.selectbox("Color sensitivity", list(SENSITIVITY_PRESETS.keys()), index=0)
        chroma_thresh = SENSITIVITY_PRESETS[sensitivity_label]
        white_thresh = st.slider("White threshold (RGB)", min_value=220, max_value=255, value=DEFAULT_WHITE_THRESH)

# Keep TWO independent pricing profiles
# SINGLE
if "single_color_slabs" not in st.session_state:
    st.session_state.single_color_slabs = DEFAULT_COLOR_SLABS.copy()
if "single_bw_slabs" not in st.session_state:
    st.session_state.single_bw_slabs = DEFAULT_BW_SLABS_SINGLE.copy()

# DUPLEX
if "duplex_color_slabs" not in st.session_state:
    st.session_state.duplex_color_slabs = DEFAULT_COLOR_SLABS.copy()
if "duplex_bw_slabs" not in st.session_state:
    st.session_state.duplex_bw_slabs = DEFAULT_BW_SLABS_DUPLEX.copy()

tab_analyze, tab_single, tab_duplex = st.tabs(["Analyze", "Pricing: Single-sided", "Pricing: Duplex"])

with tab_single:
    st.markdown("### Single-sided pricing")
    st.markdown("**Color slabs** (by *Color % of page area*). If you choose **Color printer**, even 0% color pages use the **lowest Color slab**.")
    st.session_state.single_color_slabs = st.data_editor(
        st.session_state.single_color_slabs, num_rows="dynamic", use_container_width=True,
        column_config={
            "min_pct": st.column_config.NumberColumn("Min %", min_value=0.0, max_value=100.0, step=0.5),
            "max_pct": st.column_config.NumberColumn("Max %", min_value=0.0, max_value=100.0, step=0.5),
            "price":   st.column_config.NumberColumn("Price (₹)", min_value=0.0, step=0.25),
        },
        key="single_color_editor",
    )
    st.markdown("**B/W slabs** (by *non-white %*). Used when you pick **B/W printer**.")
    st.session_state.single_bw_slabs = st.data_editor(
        st.session_state.single_bw_slabs, num_rows="dynamic", use_container_width=True,
        column_config={
            "min_pct": st.column_config.NumberColumn("Min %", min_value=0.0, max_value=100.0, step=0.5),
            "max_pct": st.column_config.NumberColumn("Max %", min_value=0.0, max_value=100.0, step=0.5),
            "price":   st.column_config.NumberColumn("Price (₹)", min_value=0.0, step=0.25),
        },
        key="single_bw_editor",
    )

    p1a, sc1 = ensure_slabs(st.session_state.single_color_slabs, "Single: Color slabs")
    p1b, sb1 = ensure_slabs(st.session_state.single_bw_slabs,   "Single: B/W slabs")
    if p1a or p1b:
        st.warning("Fix slab issues:")
        for p in (p1a + p1b): st.write("•", p)
    else:
        st.session_state.single_color_slabs = sc1
        st.session_state.single_bw_slabs = sb1
        st.success("Single-sided slabs look good.")

with tab_duplex:
    st.markdown("### Duplex pricing (used when **Back-to-back (duplex)** is selected)")
    st.markdown("**Color slabs** (by *Color % of page area*). 0% color on Color printer → lowest Color slab.")
    st.session_state.duplex_color_slabs = st.data_editor(
        st.session_state.duplex_color_slabs, num_rows="dynamic", use_container_width=True,
        column_config={
            "min_pct": st.column_config.NumberColumn("Min %", min_value=0.0, max_value=100.0, step=0.5),
            "max_pct": st.column_config.NumberColumn("Max %", min_value=0.0, max_value=100.0, step=0.5),
            "price":   st.column_config.NumberColumn("Price (₹)", min_value=0.0, step=0.25),
        },
        key="duplex_color_editor",
    )
    st.markdown("**B/W slabs** (by *non-white %*). Used when you pick **B/W printer**.")
    st.session_state.duplex_bw_slabs = st.data_editor(
        st.session_state.duplex_bw_slabs, num_rows="dynamic", use_container_width=True,
        column_config={
            "min_pct": st.column_config.NumberColumn("Min %", min_value=0.0, max_value=100.0, step=0.5),
            "max_pct": st.column_config.NumberColumn("Max %", min_value=0.0, max_value=100.0, step=0.5),
            "price":   st.column_config.NumberColumn("Price (₹)", min_value=0.0, step=0.25),
        },
        key="duplex_bw_editor",
    )

    p2a, sc2 = ensure_slabs(st.session_state.duplex_color_slabs, "Duplex: Color slabs")
    p2b, sb2 = ensure_slabs(st.session_state.duplex_bw_slabs,   "Duplex: B/W slabs")
    if p2a or p2b:
        st.warning("Fix slab issues:")
        for p in (p2a + p2b): st.write("•", p)
    else:
        st.session_state.duplex_color_slabs = sc2
        st.session_state.duplex_bw_slabs = sb2
        st.success("Duplex slabs look good.")

with tab_analyze:
    uploaded = st.file_uploader("Drag & drop a PDF", type=["pdf"], accept_multiple_files=False)
    if uploaded:
        t0 = time.perf_counter()
        with st.spinner("Analyzing pages…"):
            pages = analyze_pdf_pages(
                uploaded.getvalue(),
                dpi=DEFAULT_DPI_FAST if mode == "Fast" else DEFAULT_DPI_ACCURATE,
                white_thresh=white_thresh,
                chroma_thresh=chroma_thresh,
                ignore_border_mm=ignore_mm,
            )

            # Choose pricing set based on Sides + Printer
            using_color_slabs = (st.session_state.duplex_color_slabs if sides.startswith("Back")
                                 else st.session_state.single_color_slabs)
            using_bw_slabs    = (st.session_state.duplex_bw_slabs if sides.startswith("Back")
                                 else st.session_state.single_bw_slabs)

            df = compute_pricing(
                pages,
                printer=printer,
                color_slabs=using_color_slabs,
                bw_slabs=using_bw_slabs,
                round_per_side=True,
            )

        total_pages  = len(df)
        total_price  = float(df["price"].sum())
        total_sheets = math.ceil(total_pages / 2) if sides.startswith("Back") else total_pages
        t1 = time.perf_counter()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Pages (sides)", f"{total_pages}")
        k2.metric("Sheets", f"{total_sheets}")
        k3.metric("Total (₹)", f"{round_rupee(total_price)}")
        k4.metric("Time", f"{t1 - t0:.2f} s")

        show = df.copy()
        for col in ["white_pct", "color_pct", "black_pct"]:
            show[col] = show[col].map(lambda x: f"{x:.2f}%")
        if printer == "B/W printer":
            show["BW Ink % (used)"] = (100.0 - df["white_pct"]).map(lambda x: f"{x:.2f}%")
        show["slab_price"] = show["slab_price"].map(lambda x: f"₹{x:.2f}")
        show["price"] = show["price"].map(lambda x: f"₹{int(x)}")

        st.markdown("#### Per-side breakdown")
        st.dataframe(show.drop(columns=["bw_ink_pct"]), use_container_width=True)

        st.caption(
            ("Printer: **Color** (0% color pages use lowest Color slab)" if printer == "Color printer"
             else "Printer: **B/W** (priced by non-white %)") +
            (" • Pricing profile: **Duplex**" if sides.startswith("Back") else " • Pricing profile: **Single-sided**")
        )
    else:
        st.info("Upload a PDF to begin. Set your slabs in the pricing tabs.")
